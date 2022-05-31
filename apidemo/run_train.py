import json

from transformers import BatchEncoding, AutoTokenizer
from models.nets_span import IEToken, IESPAN, IEFromNLI
import numpy as np
import os
import torch
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names


from utils.data import get_data, get_dev_test_encodings
from utils.utils import F1MetricTag
from utils.options import parse_arguments
from utils.worker import Worker, GWorker


def create_optimizer_and_scheduler(model:torch.nn.Module, learning_rate:float, weight_decay:float, warmup_step:int, train_step:int, adam_beta1:float=0.9, adam_beta2:float=0.999, adam_epsilon:float=1e-8):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    scheduler = get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=train_step,
            )
    return optimizer, scheduler


def quick_train(opts):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if opts.seed == -1:
        import time
        opts.seed = time.time()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.gpu.count(",") > 0:
        opts.batch_size = opts.batch_size * (opts.gpu.count(",")+1)
        opts.eval_batch_size = opts.eval_batch_size * (opts.gpu.count(",")+1)
    loaders, label2id = get_data(opts)
    dev_weak_encoding, _, _ = get_dev_test_encodings(opts)
    if opts.setting == 'span':
        IEModel = IESPAN
    elif opts.setting == "token":
        IEModel = IEToken
    elif opts.setting == "nli":
        IEModel = IEFromNLI

    model = IEModel(
        hidden_dim=opts.hidden_dim,
        nclass=len(label2id) if opts.setting != "sentence" else len(label2id) - 1,
        model_name=opts.model_name,
        distributed=opts.gpu.count(",") > 0
    )
    id2label = {i:k for k, i in label2id.items()}
    if 0 not in id2label:
        id2label[0] = 'NA'

    if opts.gpu.count(",") > 0:
        model = torch.nn.DataParallel(model)
        print('para')

    model.to(torch.device('cuda:0') if torch.cuda.is_available() and (not opts.no_gpu) else torch.device('cpu'))

    if not opts.test_only:
        optimizer, scheduler = create_optimizer_and_scheduler(model, opts.learning_rate, opts.decay, opts.warmup_step, len(loaders[0]) * opts.train_epoch // opts.accumulation_steps)
    else:
        optimizer = scheduler = None

    worker = GWorker(opts) if opts.example_regularization else Worker(opts)
    worker._log(str(opts))
    worker._log(json.dumps(label2id))
    if opts.continue_train:
        worker.load(model, optimizer, scheduler)
    elif opts.test_only:
        worker.load(model)
    best_test = [None]
    best_dev = [None]
    test_metrics = None
    dev_metrics = []
    metric = "f1"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = [0]
    total_epoch = 0

    # [0,1,2,3,4,5,6,7,12,14,15] are ignored labels for comparison with zero-shot ee
    F1Metric = F1MetricTag(-100, [0,1,2,3,4,5,6,7,12,14,15][0], label2id, AutoTokenizer.from_pretrained(opts.model_name), save_dir=opts.log_dir)

    print("start training")
    while not termination:
        if not opts.test_only:
            f_loss = None
            epoch_loss, epoch_metric = worker.run_one_epoch(
                model=model,
                f_loss=f_loss,
                loader=loaders[0],
                split="train",
                optimizer=optimizer,
                scheduler=scheduler,
                metric=metric,
                max_steps=200)
            total_epoch += 1

            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}  Train Loss {epoch_loss} {epoch_metric}")
            
            dev_metrics = []
            for idev, dev_loader in enumerate(loaders[1:2]):
                _, dev_met = worker.run_one_epoch(
                    model=model,
                    loader=dev_loader,
                    split="dev",
                    metric=metric,
                    max_steps=-1,
                    collect_outputs=collect_outputs)
                if dev_met is None:
                    dev_met, dev_met_by_label = F1Metric(worker.epoch_outputs, dev_weak_encoding)
                dev_metrics.append(dev_met)
        else:

            termination = True
        
        
        dev_log = ''
        for i, dev_met in enumerate(dev_metrics):
            dev_log += f'Dev_{i} {dev_met}|' 
        for output_log in [print, worker._log]:
            output_log(
                f"Epoch {worker.epoch:3d}: {dev_log}"
            )

        if not opts.test_only:
            for i in range(len(dev_metrics)):
                if (best_dev[i] is None or dev_metrics[i] > best_dev[i]) and no_better[i] < patience:
                    best_dev[i] = dev_metrics[i]
                    if len(dev_metrics) == 1:
                        worker.save(model, optimizer, scheduler, data={'id2label':id2label},postfix=f"best")
                    else:
                        worker.save(model, optimizer, scheduler, data={'id2label':id2label},postfix=f"best.dev_{i}")
                    no_better[i] = 0
                else:
                    no_better[i] += 1
            print(f"Current: {', '.join([str(t) for t in dev_metrics])} | History Best:{', '.join([str(t) for t in best_dev])} | Patience: {no_better} : {patience}")
            if all([nb >= patience for nb in no_better]) or (worker.epoch > worker.train_epoch):
                dev_log = ''
                for i, dev_met in enumerate(best_dev):
                    dev_log += f'{dev_met.full_result},'
                for output_log in [print, worker._log]:
                    output_log(f"BEST DEV : [{dev_log}]")
                termination = True


def train(opts):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if opts.seed == -1:
        import time
        opts.seed = time.time()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.gpu.count(",") > 0:
        opts.batch_size = opts.batch_size * (opts.gpu.count(",")+1)
        opts.eval_batch_size = opts.eval_batch_size * (opts.gpu.count(",")+1)
    loaders, label2id = get_data(opts)
    dev_weak_encoding, dev_encoding, test_encoding = get_dev_test_encodings(opts)
    if dev_weak_encoding is None:
        dev_encoding = [dev_encoding]
    else:
        dev_encoding = [dev_weak_encoding, dev_encoding]
    if opts.setting == 'span':
        IEModel = IESPAN
    elif opts.setting == "token":
        IEModel = IEToken
    elif opts.setting == "nli":
        IEModel = IEFromNLI

    model = IEModel(
        hidden_dim=opts.hidden_dim,
        nclass=len(label2id) if opts.setting != "sentence" else len(label2id) - 1,
        model_name=opts.model_name,
        distributed=opts.gpu.count(",") > 0
    )

    if opts.gpu.count(",") > 0:
        model = torch.nn.DataParallel(model)
        print('para')

    model.to(torch.device('cuda:0') if torch.cuda.is_available() and (not opts.no_gpu) else torch.device('cpu'))

    if not opts.test_only:
        optimizer, scheduler = create_optimizer_and_scheduler(model, opts.learning_rate, opts.decay, opts.warmup_step, len(loaders[0]) * opts.train_epoch // opts.accumulation_steps)
    else:
        optimizer = scheduler = None

    worker = GWorker(opts) if opts.example_regularization else Worker(opts)
    worker._log(str(opts))
    worker._log(json.dumps(label2id))
    if opts.continue_train:
        worker.load(model, optimizer, scheduler)
    elif opts.test_only:
        worker.load(model)
    best_test = [None for _ in range(len(loaders)-2)]
    best_dev = [None for _ in range(len(loaders)-2)]
    test_metrics = None
    dev_metrics = []
    metric = "f1"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = [0 for _ in range(len(loaders)-2)]
    total_epoch = 0

    # [0,1,2,3,4,5,6,7,12,14,15] are ignored labels for comparison with zero-shot ee
    F1Metric = F1MetricTag(-100, [0,1,2,3,4,5,6,7,12,14,15][0], label2id, AutoTokenizer.from_pretrained(opts.model_name), save_dir=opts.log_dir)

    print("start training")
    while not termination:
        if not opts.test_only:
            f_loss = None
            epoch_loss, epoch_metric = worker.run_one_epoch(
                model=model,
                f_loss=f_loss,
                loader=loaders[0],
                split="train",
                optimizer=optimizer,
                scheduler=scheduler,
                metric=metric,
                max_steps=-200)
            total_epoch += 1

            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}  Train Loss {epoch_loss} {epoch_metric}")
            
            dev_metrics = []
            for idev, dev_loader in enumerate(loaders[1:-1]):
                _, dev_met = worker.run_one_epoch(
                    model=model,
                    loader=dev_loader,
                    split="dev",
                    metric=metric,
                    max_steps=-1,
                    collect_outputs=collect_outputs)
                if dev_met is None:
                    dev_met, dev_met_by_label = F1Metric(worker.epoch_outputs, dev_encoding[idev])
                dev_metrics.append(dev_met)
        else:

            termination = True
        
        _, test_metrics = worker.run_one_epoch(
            model=model,
            loader=loaders[-1],
            split="test",
            metric=metric,
            collect_outputs=collect_outputs)
        test_outputs = worker.epoch_outputs
        if test_metrics is None:
            test_metrics, test_metrics_by_label = F1Metric(worker.epoch_outputs, test_encoding)
        
        dev_log = ''
        for i, dev_met in enumerate(dev_metrics):
            dev_log += f'Dev_{i} {dev_met}|' 
        for output_log in [print, worker._log]:
            output_log(
                f"Epoch {worker.epoch:3d}: {dev_log}"
                f"Test {test_metrics.full_result}"
            )
        print(test_metrics_by_label)
        macro = sum([v[2] * 2 / max(v[0] + v[1],1) for v in test_metrics_by_label.values()]) / len(test_metrics_by_label)
        print(macro)

        if not opts.test_only:
            for i in range(len(dev_metrics)):
                if (best_dev[i] is None or dev_metrics[i] > best_dev[i]) and no_better[i] < patience:
                    best_dev[i] = dev_metrics[i]
                    if len(dev_metrics) == 1:
                        worker.save(model, optimizer, scheduler, postfix=f"best.{opts.run_fold}")
                    else:
                        worker.save(model, optimizer, scheduler, postfix=f"best.{opts.run_fold}.dev_{i}")
                    best_test[i] = test_metrics
                    no_better[i] = 0
                else:
                    no_better[i] += 1
            print(f"Current: {', '.join([str(t) for t in dev_metrics])} | History Best:{', '.join([str(t) for t in best_dev])} | Patience: {no_better} : {patience}")
            if (worker.epoch+1) % 10 == 0:
                worker.save(model, optimizer, scheduler, postfix=str((worker.epoch+1) // 10))
            if all([nb >= patience for nb in no_better]) or (worker.epoch > worker.train_epoch):
                dev_log = ''
                for i, dev_met in enumerate(best_dev):
                    dev_log += f'{dev_met.full_result},'
                test_log = ''
                for i, test_met in enumerate(best_test):
                    test_log += f'{test_met.full_result},'
                for output_log in [print, worker._log]:
                    output_log(f"BEST DEV : [{dev_log}]")
                    output_log(f"BEST TEST: [{test_log}]")
                termination = True

def main():
    opts = parse_arguments()
    train(opts)
    


if __name__ == "__main__":
    main()