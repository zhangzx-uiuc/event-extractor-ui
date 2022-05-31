from argparse import ArgumentParser
from collections import Counter
import json
from multiprocessing import Process, Queue, Value
import numpy as np
import os
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, BatchEncoding, AutoTokenizer, AutoModelForMaskedLM, AutoModel, PreTrainedModel
from typing import *
from utils.utils import WSAnnotations, get_dataset_and_loader, F1MetricTag, preprocess_func_token
from utils.options import parse_arguments_for_weak_supervision
import importlib

os.environ["TOKENIZERS_PARALLELISM"] = "true"

Tokenized = Tuple[str, List[str], List[str], List[str]] # document_id, tokens, pos_tags and lemmas
Document = Tuple[str, str] # document_id, text
Keyword = Union[str, Tuple[str, str]]
Words = Union[List[Keyword], Set[Keyword]] # list or set of words
Occurence = Tuple[str, int] # document_id, token_offset
IndexOccurence = Tuple[Union[int, str], Union[int, Tuple[int, int]]] # first item either be an integer or an str that is an integer
Offset = Tuple[int, int]
Annotation = Tuple[int, int, str]
Annotations = List[Annotation]

def create_multiple_processes_with_progress(target_func, nprocess, inputs, *args)->List[Any]:
    '''
    target_func: function of individual processes, first four arguments being process_id, output_queue, progress_value, inputs
    '''
    processes = []
    ninstance_per_process = (len(inputs) + nprocess - 1) // nprocess
    queue = Queue(nprocess + 1)
    progress = Value('i', 0)
    current_progress = 0
    print("start working")
    for i in range(nprocess):
        sub_inputs = inputs[ninstance_per_process * i: ninstance_per_process * (i+1)]
        if len(sub_inputs) > 0:
            p = Process(target=target_func, args=(i, queue, progress, sub_inputs, *args))
            p.start()
            processes.append(p)
    outputs = [None for _ in range(len(processes))]
    progress_bar = tqdm(total=len(inputs))
    while True:
        running = any(p.is_alive() for p in processes)
        if progress.value > current_progress:
            progress_bar.update(progress.value - current_progress)
            current_progress = progress.value
        while not queue.empty():
            output = queue.get()
            outputs[output["id"]] = output["output"]
        if not running:
            break
    progress_bar.update(progress.value - current_progress)
    progress_bar.refresh()
    for p in processes:
        p.join()
    if isinstance(outputs[0], list):
        outputs = [t for t in outputs if t is not None]
        outputs = [tt for t in outputs for tt in t]
    progress_bar.close()
    for p in processes:
        p.terminate()
    return outputs


def dist_with_progress(step_func:Callable, batch_size:int=1, pre_func:Optional[Callable]=None, post_func:Optional[Callable]=None, break_batch_output:bool=True)->Callable:

    if batch_size == 1:
        def epoch_func(id:int, queue:Queue, progress:Value, dataset:List[Any], *args, **kwargs):
            if pre_func is not None:
                step_kwargs, post_kwargs = pre_func(id, *args, **kwargs)
            else:
                step_kwargs = post_kwargs = None
            outputs = []
            for data in dataset:
                if step_kwargs is None:
                    output = step_func(data, *args, **kwargs)
                else:
                    output = step_func(data, **step_kwargs)
                outputs.append(output)
                progress.value = progress.value + 1
            if post_func is not None:
                if post_kwargs is not None:
                    outputs = post_func(id, outputs, **post_kwargs)
                else:
                    outputs = post_func(id, outputs, *args, **kwargs)
            queue.put({"id":id, "output": outputs})
    else:
        def epoch_func(id:int, queue:Queue, progress:Value, dataset:List[Any], *args, **kwargs):
            if pre_func is not None:
                step_kwargs, post_kwargs = pre_func(id, *args, **kwargs)
            else:
                step_kwargs = post_kwargs = None
            outputs = []
            for ibatch in range(0, len(dataset), batch_size):
                batch = dataset[ibatch:ibatch+batch_size]
                if step_kwargs is None:
                    batch_output = step_func(batch, *args, **kwargs)
                else:
                    batch_output = step_func(batch, **step_kwargs)
                if isinstance(batch_output, list) and break_batch_output:
                    outputs.extend(batch_output)
                else:
                    outputs.append(batch_output)
                progress.value = progress.value + len(batch)
            if post_func is not None:
                if post_kwargs is not None:
                    outputs = post_func(id, outputs, **post_kwargs)
                else:
                    outputs = post_func(id, outputs, *args, **kwargs)
            queue.put({"id":id, "output": outputs})

    return epoch_func


def pack_arguments_as_list_wrapper(func)->Callable:
    def _func(args:List):
        return func(*args)
    return _func


def _preprocess_func(input_file):
    with open(input_file, "rt") as fp:
        data = json.load(fp)
    data_sentences = [(str(i), t) for i, t in enumerate(data)]
    return data_sentences


def _clean_keywords(label_keywords:Dict[str, List[str]]) -> Dict[str, List[str]]:
    all_keywords = Counter([keyword for keywords in label_keywords.values() for keyword in keywords]).most_common()
    discard_keywords = {keyword for keyword, nlabel in all_keywords if nlabel > 1}
    return {label: [keyword for keyword in keywords if keyword not in discard_keywords] for label, keywords in label_keywords.items()}


def _clean_keywords2(label_keywords:Dict[str, List[str]]) -> Dict[str, List[str]]:
    lk = {}
    kused = set()
    for l, k in label_keywords.items():
        lk[l] = []
        for kw in k:
            if kw in kused: continue
            lk[l].append(kw)
            kused.add(kw)
    return lk


def _prepare_mask_inputs(encodings:List[BatchEncoding], offsets:List[Offset], tokenizer:PreTrainedTokenizerFast, is_word_offsets:bool=True)->BatchEncoding:
    input_ids = []
    attention_mask = []
    if is_word_offsets:
        for encoding, offset in zip(encodings, offsets):
            start = encoding.word_to_tokens(offset[0])
            end = encoding.word_to_tokens(offset[1])
            if start is None:
                print(offset, encoding)
                continue
            else:
                start = start.start
                if end is None:
                    end = start + 1
                else:
                    end = end.start
            input_ids.append(encoding.input_ids[:start] + [tokenizer.mask_token_id] + encoding.input_ids[end:])
            attention_mask.append(encoding.attention_mask[:start] + [1] + encoding.attention_mask[end:])
    else:
        for encoding, offset in zip(encodings, offsets):
            start = encoding.char_to_tokens(offset[0]).start
            end = encoding.char_to_tokens(offset[1]).start
            input_ids.append(encoding.input_ids[:start] + [tokenizer.mask_token_id] + encoding.input_ids[end:])
            attention_mask.append(encoding.attention_mask[:start] + [1] + encoding.attention_mask[end:])
    batch_encoding = tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask}, return_attention_mask=True, return_tensors='pt', max_length=512, padding='longest')
    return batch_encoding


def _prepare_span_inputs(encodings:List[BatchEncoding], annotations:List[List[Offset]], tokenizer:PreTrainedTokenizerFast, is_word_offsets:bool=True) -> Tuple[BatchEncoding, torch.FloatTensor, List[str]]:
    token_offsets = []
    keywords = []
    if is_word_offsets:
        for idx, (encoding, annotation) in enumerate(zip(encodings, annotations)):
            for offset in annotation:
                start = encoding.word_to_tokens(offset[0])
                end = encoding.word_to_tokens(offset[1])
                if start is None:
                    continue
                else:
                    start = start.start
                    if end is None:
                        end = start + 1
                    else:
                        end = end.start
                token_offsets.append((idx, start, end))
                keywords.append(offset[2])
    else:
        for idx, (encoding, annotation) in enumerate(zip(encodings, annotations)):
            for offset in annotation:
                start = encoding.char_to_tokens(offset[0])
                end = encoding.char_to_tokens(offset[1])
                if start is None:
                    continue
                else:
                    start = start.start
                    if end is None:
                        end = start + 1
                    else:
                        end = end.start
                token_offsets.append((idx, start, end))
                keywords.append(offset[2])
    batch_encoding = tokenizer.pad(encodings, return_attention_mask=True, return_tensors='pt', max_length=512, padding='longest')
    batch_size = batch_encoding.input_ids.size(0)
    seq_length = batch_encoding.input_ids.size(1)
    span_mask = torch.zeros(len(token_offsets), batch_size * seq_length)
    for i, (j, start, end) in enumerate(token_offsets):
        span_mask[i, j*seq_length+start:j*seq_length+end] = 1. / (end - start)
    return batch_encoding, span_mask, keywords


def _tokenize_and_lemmatize_corpus(text_corpus: List[Document]) -> List[Tokenized]:
    lemmatizer = WordNetLemmatizer()
    for doc_id, doc in tqdm(text_corpus):
        tokens = pos_tag(word_tokenize(doc), tagset='universal')
        lemmas = [lemmatizer.lemmatize(token, 'v') if pos == 'VERB' else lemmatizer.lemmatize(token, 'n') for token, pos in tokens]
        return {
            "tokens": [token for token, _ in tokens],
            "pos_tags": [pos for _, pos in tokens],
            "lemmas": lemmas
            }


def _dist_tokenize_and_lemmatize_corpus() -> Callable:
    def _pre_doc(*args, **kwargs):
        return {'lemmatizer': WordNetLemmatizer()}, None
    def _step_doc(doc, lemmatizer):
        if isinstance(doc[1], str):
            tokens = pos_tag(word_tokenize(doc[1].lower()), tagset='universal')
        else:
            tokens = pos_tag([t.lower() for t in doc[1]], tagset='universal')
        lemmas = [lemmatizer.lemmatize(token, 'v') if pos == 'VERB' else lemmatizer.lemmatize(token, 'n') for token, pos in tokens]
        return {
            "tokens": [token for token, _ in tokens],
            "pos_tags": [pos for _, pos in tokens],
            "lemmas": lemmas,
            "sentence": doc[1]
            }
    return dist_with_progress(_step_doc, 1, _pre_doc)


def _find_keyword_occurences_in_corpus(tokenized_corpus: List[Tokenized], keywords:Words) -> Dict[str, List[Occurence]]:
    occurences = {k: [] for k in keywords}
    keyword_constraints = {k.split(".")[0]: k.split(".")[1] for k in keywords if k.count('.') > 0}
    for idx, tokenized in enumerate(tqdm(tokenized_corpus)):
        for position, token in enumerate(tokenized[1]):
            if position < len(tokenized[3]) - 1 and f"{token} {tokenized[3][position+1]}" in occurences:
                occurences[f"{token} {tokenized[3][position+1]}"].append((tokenized[0], position, 1))
            if token in occurences:
                occurences[token].append((tokenized[0], position))
            elif f"{token}.{tokenized[2][position][0].lower()}" in occurences:
                occurences[f"{token}.{tokenized[2][position][0].lower()}"].append((tokenized[0], position))
            else:
                lemma = tokenized[3][position]
                if position < len(tokenized[3]) - 1 and f"{lemma} {tokenized[3][position+1]}" in occurences:
                    occurences[f"{lemma} {tokenized[3][position+1]}"].append((tokenized[0], position, 1))
                if lemma in occurences:
                    occurences[lemma].append((tokenized[0], position))
                elif f"{lemma}.{tokenized[2][position][0].lower()}" in occurences:
                    occurences[f"{lemma}.{tokenized[2][position][0].lower()}"].append((tokenized[0], position))
    return occurences


def _dist_find_keyword_occurences_in_corpus(id:int, queue:Queue, progress:Value, tokenized_corpus: List[Tokenized], keywords:Words) -> None:
    occurences = {k: [] for k in keywords}
    for idx, tokenized in enumerate(tokenized_corpus):
        for position, lemma in enumerate(tokenized[3]):
            if lemma in occurences:
                occurences[lemma].append((tokenized[0], position))
        progress.value = progress.value + 1

    queue.put({"id": id, "output": occurences})


def _prepare_document_for_pretrained_language_model(tokenizer:PreTrainedTokenizerFast, tokenized:Tokenized) -> BatchEncoding:
    tokens = tokenized[1]
    encoding = tokenizer(tokens, add_special_tokens=True, is_split_into_words=True)
    return encoding


def _create_tokenizer_and_process_corpus(tokenized_corpus:List[Tokenized], model_name:str) -> List[BatchEncoding]:
    if model_name.lower().startswith('roberta'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = []
    for tokenized_document in tokenized_corpus:
        encodings.append(_prepare_document_for_pretrained_language_model(tokenizer, tokenized_document))
    return encodings


def _dist_create_tokenizer_and_process_corpus(id:int, queue:Queue, progress:Value, tokenized_corpus:List[Tokenized], model_name:str)-> None:
    if model_name.lower().startswith('roberta'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = []
    for tokenized in tokenized_corpus:
        tokens = tokenized[1]
        encoding = tokenizer(
            tokens,
            add_special_tokens=True,
            is_split_into_words=True,
            padding='longest',
            max_length=512,
            truncation=True)
        encodings.append(encoding)
        progress.value = progress.value + 1
    queue.put({"id": id, "output": encodings})


def _keyword_representation(id:int, encodings:List[Tuple[BatchEncoding, Annotations]], model_name:Optional[str]=None, tokenizer:Optional[PreTrainedTokenizerFast]=None, model:Optional[PreTrainedModel]=None, batch_size:int=16) -> Dict[str, Tuple[int, List[float]]]:
    with torch.no_grad():
        if model_name is not None:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            if model is None:
                model = AutoModel.from_pretrained(model_name)
        elif model is None or tokenizer is None:
            raise ValueError("need either model name or loaded model/tokenizer")
        model = model.eval()
        model = model.cuda(f"cuda:{id}")
        keyword_vectors = {}
        for ibatch in tqdm(range(0, len(encodings), batch_size)):
            batch = encodings[ibatch:ibatch+batch_size]
            batch_encodings = [t[0] for t in batch]
            batch_annotation = [t[1] for t in batch]
            batch_encoding, span_mask, batch_keywords = _prepare_span_inputs(batch_encodings, batch_annotation, tokenizer)
            output = model(**batch_encoding.to(f"cuda:{id}")).last_hidden_state.flatten(end_dim=1)
            output = torch.matmul(span_mask.cuda(f"cuda:{id}"), output).detach().cpu()
            for i, keyword in enumerate(batch_keywords):
                if keyword not in keyword_vectors:
                    keyword_vectors[keyword] = (1, output[i])
                else:
                    nvector, representation = keyword_vectors[keyword]
                    keyword_vectors[keyword] = (nvector + 1, (representation * nvector + output[i] ) / (nvector + 1))
        keyword_vectors = {k: (v[0], v[1].numpy()) for k,v in keyword_vectors.items()}
    return keyword_vectors


def _dist_keyword_representation(id:int, queue:Queue, progress:Value, encodings:List[Tuple[BatchEncoding, Annotations]], model_name:str, batch_size:int=16) -> None:
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.eval()
        model = model.cuda(f"cuda:{id}")
        keyword_vectors = {}
        for ibatch in range(0, len(encodings), batch_size):
            batch = encodings[ibatch:ibatch+batch_size]
            batch_encodings = [t[0] for t in batch]
            batch_annotation = [t[1] for t in batch]
            batch_encoding, span_mask, batch_keywords = _prepare_span_inputs(batch_encodings, batch_annotation, tokenizer)
            output = model(**batch_encoding.to(f"cuda:{id}")).last_hidden_state.flatten(end_dim=1)
            output = torch.matmul(span_mask.cuda(f"cuda:{id}"), output).detach().cpu()
            for i, keyword in enumerate(batch_keywords):
                if keyword not in keyword_vectors:
                    keyword_vectors[keyword] = (1, output[i])
                else:
                    nvector, representation = keyword_vectors[keyword]
                    keyword_vectors[keyword] = (nvector + 1, (representation * nvector + output[i] ) / (nvector + 1))
            progress.value = progress.value + len(batch_encodings)
        keyword_vectors = {k: (v[0], v[1].tolist()) for k,v in keyword_vectors.items()}
    queue.put({"id": id, "output": keyword_vectors})


def _mask_prediction(id:int, encodings:List[BatchEncoding], occurences:List[IndexOccurence], model_name:Optional[str]=None, tokenizer:Optional[PreTrainedTokenizerFast]=None, mlm:Optional[PreTrainedModel]=None, batch_size:int=16, top_k:int=50) -> List[List[str]]:
    if model_name is not None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if mlm is None:
            mlm = AutoModelForMaskedLM.from_pretrained(model_name)
    elif mlm is None or tokenizer is None:
        raise ValueError("need either model name or loaded model/tokenizer")
    mlm.eval()
    mlm = mlm.cuda(f"cuda:{id}")
    prediction = []
    with torch.no_grad():
        for ibatch in tqdm(range(0, len(occurences), batch_size)):
            offsets = []
            encoded = []
            for occ in occurences[ibatch:ibatch+batch_size]:
                if len(occ) == 2:
                    i, j = occ; k = j + 1
                else:
                    i, j, _ = occ; k = j + 2
                offsets.append((j, k))
                encoded.append(encodings[int(i)])
            batch_encodings= _prepare_mask_inputs(encoded, offsets, tokenizer).to("cuda:0")
            logits = mlm(**batch_encodings).logits[batch_encodings["input_ids"]==tokenizer.mask_token_id]
            _, topk = logits.topk(top_k, dim=-1)
            prediction.extend([tokenizer.convert_ids_to_tokens(t) for t in topk])
    del mlm, tokenizer
    return prediction


def _filter_occurences(keyword_occurence_prediction: List[Tuple[str, IndexOccurence, List[str]]], keyword_labels: Dict[str, str], nthreshold:Union[Dict[str, int], int]=1) -> List[Tuple[str, IndexOccurence]]:
    lemmatizer = WordNetLemmatizer()
    outputs = []
    if isinstance(nthreshold, int):
        nthreshold = {keyword: nthreshold for keyword in keyword_labels}
    for keyword, occurences, prediction in keyword_occurence_prediction:
        n = 0
        for word in prediction:
            if word in keyword_labels and keyword_labels[keyword] == keyword_labels[word]:
                n += 1
            else:
                word_v = lemmatizer.lemmatize(word, 'v')
                if word_v in keyword_labels and keyword_labels[keyword] == keyword_labels[word_v]:
                    n += 1
                else:
                    word_n = lemmatizer.lemmatize(word, 'n')
                    if word_n in keyword_labels and keyword_labels[keyword] == keyword_labels[word_n]:
                        n += 1
            if n == nthreshold[keyword]:
                outputs.append((occurences, keyword))
                break
    return outputs


def _dist_filter_occurences(id:int, queue:Queue, progress:Value, keyword_occurence_prediction: List[Tuple[str, IndexOccurence, List[str]]], keyword_labels: Dict[str, str], nthreshold:Union[Dict[str, int], int]=1) -> None:
    def _match_label(kw_label, sb_label):
        if kw_label.startswith("Justice"):
            return kw_label.split(":")[0] == sb_label.split(":")[0]
        else:
            return kw_label == sb_label
    keyword_labels = keyword_labels.copy()
    update_keywords = {}
    for keyword in keyword_labels:
        if "." in keyword:
            update_keywords[keyword.split(".")[0]] = keyword_labels[keyword]
    keyword_labels.update(update_keywords)

    lemmatizer = WordNetLemmatizer()
    outputs = []
    if isinstance(nthreshold, int):
        nthreshold = {keyword: nthreshold for keyword in keyword_labels}
    for keyword, occurences, prediction in keyword_occurence_prediction:
        n = 0
        for word in prediction:
            if word == keyword.split(".")[0]:
                continue
            if word in keyword_labels and _match_label(keyword_labels[keyword], keyword_labels[word]):
                n += 1
            else:
                word_v = lemmatizer.lemmatize(word, 'v')
                if word_v in keyword_labels and _match_label(keyword_labels[keyword], keyword_labels[word_v]):
                    n += 1
                else:
                    word_n = lemmatizer.lemmatize(word, 'n')
                    if word_n in keyword_labels and _match_label(keyword_labels[keyword], keyword_labels[word_n]):
                        n += 1
            if n >= nthreshold[keyword]:
                outputs.append((occurences, keyword))
                break
        progress.value = progress.value + 1
    queue.put({"id": id, "output": outputs})


# def keyword_matching(id:int, encodings:List[BatchEncoding], keyword_vectors:torch.FloatTensor, keywordid2label:Dict[int, str], label2id:Dict[str, int], th:float, uth:float, data_sentences:Optional[List[Document]]=None, model_name:Optional[str]=None, tokenizer:Optional[PreTrainedTokenizerFast]=None, model:Optional[PreTrainedModel]=None, batch_size:int=16):
def keyword_matching(id:int, encodings:List[BatchEncoding], keyword_vectors:torch.FloatTensor, keywordid2label:Dict[int, str], label2id:Dict[str, int], th:float, uth:float, data_sentences:Optional[List[Document]]=None, model_name:Optional[str]=None, tokenizer:Optional[PreTrainedTokenizerFast]=None, model:Optional[PreTrainedModel]=None, batch_size:int=16):
    if model_name is not None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            model = AutoModel.from_pretrained(model_name)
    elif model is None or tokenizer is None:
        raise ValueError("need either model name or loaded model/tokenizer")
    model = model.cuda(f"cuda:{id}")
    print(th)
    annotator = WSAnnotations(threshold=th, tokenizer=tokenizer, label2id=label2id, id2label=keywordid2label, uthreshold=uth)
    with torch.no_grad():
        keyword_vectors = keyword_vectors.cuda(f"cuda:{id}")
        model.eval()
        outputs = []
        if data_sentences is None: niter = len(encodings)
        else: niter = len(data_sentences)
        for ibatch in tqdm(range(0, niter, batch_size)):
            if data_sentences is None:
                batch = encodings[ibatch:ibatch+batch_size]
                batch_encoding = tokenizer.pad(batch, return_tensors='pt')
            else:
                batch = data_sentences[ibatch:ibatch+batch_size]
                batch = [t[1] for t in batch]
                if isinstance(batch[0], list):
                    batch = [' '.join(t) for t in batch]
                batch_encoding = tokenizer(
                    batch,
                    add_special_tokens=True,
                    return_tensors='pt',
                    padding='longest',
                    truncation=True,
                    max_length=512)
            batch_encoding_gpu = batch_encoding.to(keyword_vectors.device)
            output = model(**batch_encoding_gpu).last_hidden_state
            output = output / torch.norm(output, dim=-1, keepdim=True)
            score = torch.matmul(output, keyword_vectors.transpose(0, 1)).detach().cpu()
            if data_sentences is None:
                input_ids = batch_encoding.input_ids.detach().cpu()
                batch_annotations = annotator.annotate(input_ids, score)
            else:
                batch_annotations = annotator.annotate(batch_encoding.to(torch.device('cpu')), score)
            outputs.extend(batch_annotations)
    if outputs[0]['tokens'] is None:
        for (doc_id, sent), ann in zip(data_sentences, outputs):
            ann['tokens'] = sent
    return outputs
    # if model_name is not None:
    #     if tokenizer is None:
    #         tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     if model is None:
    #         model = AutoModel.from_pretrained(model_name)
    # elif model is None or tokenizer is None:
    #     raise ValueError("need either model name or loaded model/tokenizer")
    # model = model.cuda(f"cuda:{id}")
    # print(th)
    # annotator = WSAnnotations(threshold=th, tokenizer=tokenizer, label2id=label2id, id2label=keywordid2label, uthreshold=uth)
    # with torch.no_grad():
    #     keyword_vectors = keyword_vectors.cuda(f"cuda:{id}")
    #     model.eval()
    #     outputs = []
    #     if data_sentences is None: niter = len(encodings)
    #     else: niter = len(data_sentences)
    #     for ibatch in tqdm(range(0, niter, batch_size)):
    #         if data_sentences is None:
    #             batch = encodings[ibatch:ibatch+batch_size]
    #             batch_encoding = tokenizer.pad(batch, return_tensors='pt')
    #         else:
    #             batch = data_sentences[ibatch:ibatch+batch_size]
    #             batch = [t[1] for t in batch]
    #             if isinstance(batch[0], list):
    #                 batch = [' '.join(t) for t in batch]
    #             batch_encoding = tokenizer(
    #                 batch,
    #                 add_special_tokens=True,
    #                 return_tensors='pt',
    #                 padding='longest',
    #                 truncation=True,
    #                 max_length=512)
    #         batch_encoding_gpu = batch_encoding.to(keyword_vectors.device)
    #         output = model(**batch_encoding_gpu).last_hidden_state
    #         output = output / torch.norm(output, dim=-1, keepdim=True)
    #         score = torch.matmul(output, keyword_vectors.transpose(0, 1)).detach().cpu()
    #         if data_sentences is None:
    #             input_ids = batch_encoding.input_ids.detach().cpu()
    #             batch_annotations = annotator.annotate(input_ids, score)
    #         else:
    #             batch_annotations = annotator.annotate(batch_encoding.to(torch.device('cpu')), score)
    #         outputs.extend(batch_annotations)
    # if outputs[0]['tokens'] is None:
    #     for (doc_id, sent), ann in zip(data_sentences, outputs):
    #         ann['tokens'] = sent
    # return outputs


def evaluate_vectors_on_corpus(threshold:float, keyword_vectors:torch.FloatTensor, label2id:Dict[str, int], corpus_path:Optional[str]=None, model_name:Optional[str]=None, method:str="token", keywordid2labelid:Optional[Dict[int,int]]=None, model:Optional[PreTrainedModel]=None, dataloader:Optional[torch.utils.data.DataLoader]=None) -> Dict[str, float]:

    if method == "token":
        metric = F1MetricTag(pad_value=-100, ignore_labels=0, label2id=label2id, save_dir=None)
    else:
        metric = None
    if dataloader is None:
        _, dataloader = get_dataset_and_loader(corpus_path, method, model_name)
    with torch.no_grad():
        if model is None:
            model = AutoModel.from_pretrained(model_name)
        model = model.cuda()
        keyword_vectors = keyword_vectors.cuda()
        nlabel = 0
        npred = 0
        nmatch = 0
        outputs = {"prediction": [], "label": []}
        for batch in tqdm(dataloader):
            hidden = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda()).last_hidden_state
            if method == "token":
                score = torch.matmul(hidden / torch.norm(hidden, dim=-1, keepdim=True), keyword_vectors.transpose(0, 1))
                val, pred = torch.max(score, dim=-1)
                pred = pred + 1
                pred[val < threshold] = 0
                if keywordid2labelid is not None:
                    for kid, lid in keywordid2labelid.items():
                        pred[pred==kid] = lid
                pred_npy = pred.detach().cpu().numpy()
                label_npy = batch['labels'].numpy()
                outputs['prediction'].append(pred_npy)
                outputs['label'].append(label_npy)
            else:
                raise NotImplementedError
    result = metric(outputs)
    return {"precision": result[0].full_result[0], "recall": result[0].full_result[1], "f1": result[0].full_result[2], "others": result[1]}


def cluster_keyword_vectors(keyword_vectors: np.ndarray, label_vectors: np.ndarray, keywordid2labelid: np.ndarray, threshold:float=0.75) -> Dict[int, Dict[str, Union[np.ndarray, float, int]]]:
    assert keyword_vectors.shape[0] == keywordid2labelid.shape[0]
    assert len(keywordid2labelid.shape) == 1
    # from spherecluster import SphericalKMeans
    from utils.spherical_kmeans import SphericalKMeans
    label_clusters = {}
    for label_id in range(np.min(keywordid2labelid), np.max(keywordid2labelid)+1):
        label_keywords = keywordid2labelid == label_id
        label_keyword_ids = np.nonzero(label_keywords)[0]
        if len(label_keyword_ids) == 0:
            print(label_id)
        label_keyword_vectors = keyword_vectors[label_keywords]
        cosine_with_mean = np.matmul(label_keyword_vectors, label_vectors[label_id])
        farthest = [np.min(cosine_with_mean)]
        center = [label_vectors[label_id]]
        prediction = [[(id_, 0) for id_ in label_keyword_ids]]
        if farthest[0] < threshold:
            for ncluster in range(2, min(8, label_keyword_vectors.shape[0])+1):
                kmeans = SphericalKMeans(ncluster, n_init=32)
                kmeans.fit(label_keyword_vectors.astype(np.float32))
                center.append(kmeans.cluster_centers_)
                farthest_val = 2
                for icluster in range(ncluster):
                    cluster_keyword_vectors = label_keyword_vectors[kmeans.labels_ == icluster]
                    cosine_with_mean = np.matmul(cluster_keyword_vectors, kmeans.cluster_centers_[icluster])
                    if np.min(cosine_with_mean) < farthest_val:
                        farthest_val = np.min(cosine_with_mean)
                farthest.append(farthest_val)
                prediction.append([(label_keyword_ids[i], kmeans.labels_[i]) for i in range(len(label_keyword_ids))])
                if farthest_val >= threshold:
                    break
            farthest = np.array(farthest)
            if np.sum((farthest >= threshold).astype(np.int64)) == 0:
                scluster = np.argmax(farthest) + 1
            else:
                scluster = len(farthest)
        else:
            scluster = 1

        label_clusters[label_id] = {
            "ncluster": scluster,
            "cosine": farthest[scluster-1],
            "center": center[scluster-1],
            "prediction": prediction[scluster-1]
        }
        print(label_id, prediction[scluster-1])
    return label_clusters


def generate(train_file:str, dev_file:str, label_json:str, example_json:str, model_name, output_save_dir:str, corpus_jsonl:str, no_evaluate:bool=False, threshold:float=0., force:bool=False):
    model = None
    tokenizer = None

    if (example_json == "" or no_evaluate) and threshold <= 0:
        raise ValueError("Need to provide either example data file to determine threshold value or threshold value")

    print("tokenizing and lemmatizing corpus")
    if force or not os.path.exists(corpus_jsonl):
        _pfunc = _preprocess_func
        data_sentences = _pfunc(train_file)
        corpus = create_multiple_processes_with_progress(_dist_tokenize_and_lemmatize_corpus(), 8, data_sentences)
        for data in corpus: data["split"] = "train"
        if dev_file != "":
            dev_data_sentences = _pfunc(dev_file)
            dev_corpus = create_multiple_processes_with_progress(_dist_tokenize_and_lemmatize_corpus(), 8, dev_data_sentences)
            for data in dev_corpus: data["split"] = "dev"
        else:
            dev_data_sentences = []
            dev_corpus = []
        with open(corpus_jsonl, "wt") as fp:
            for data in corpus + dev_corpus:
                fp.write(json.dumps(data)+"\n")
        corpus = [(f"{idx}", data["tokens"], data["pos_tags"], data["lemmas"]) for idx, data in enumerate(corpus)]
        dev_corpus = [(f"{idx}", data["tokens"], data["pos_tags"], data["lemmas"]) for idx, data in enumerate(dev_corpus)]
    else:
        with open(corpus_jsonl, "rt") as fp:
            corpus = []
            data_sentences = []
            dev_corpus = []
            dev_data_sentences = []
            for idx, line in enumerate(fp):
                data = json.loads(line)
                if 'sentence' not in data:
                    data['sentence'] = " ".join(data['tokens'])
                if data["split"] == "train":
                    corpus.append((f"{len(corpus)}", data["tokens"], data["pos_tags"], data["lemmas"]))
                    data_sentences.append((f"{len(data_sentences)}", data['sentence']))
                else:
                    dev_corpus.append((f"{len(dev_corpus)}", data["tokens"], data["pos_tags"], data["lemmas"]))
                    dev_data_sentences.append((f"{len(dev_data_sentences)}", data['sentence']))

    print("searching keyword in corpus")
    with open(label_json, "rt") as fp:
        label_info = json.load(fp)
        label2id = {label: info["id"] for label, info in label_info.items()}; label2id["NA"] = 0
        id2label = {v:k for k,v in label2id.items()}
        label_keywords = _clean_keywords2({label:list(set(info["seed_keywords"])) for label, info in label_info.items()})
        keyword_labels = {vv:k for k,v in label_keywords.items() for vv in v}
        keywords = [keyword for keywords in label_keywords.values() for keyword in keywords]
    if force or not os.path.exists(os.path.join(output_save_dir, f"keyword_occurence.json")):
        keyword_occurences = _find_keyword_occurences_in_corpus(corpus, keywords)
        with open(os.path.join(output_save_dir, f"keyword_occurence.json"), "wt") as fp:
            json.dump(keyword_occurences, fp, indent=4)
    else:
        with open(os.path.join(output_save_dir, f"keyword_occurence.json"), "rt") as fp:
            keyword_occurences = json.load(fp)

    print("prepare corpus for bert")
    if force or not os.path.exists(os.path.join(output_save_dir, f"{model_name}-encodings.th")):
        prepared_corpus = create_multiple_processes_with_progress(_dist_create_tokenizer_and_process_corpus, 8, corpus, model_name)
        torch.save(prepared_corpus, os.path.join(output_save_dir, f"{model_name}-encodings.th"))
    else:
        prepared_corpus = torch.load(os.path.join(output_save_dir, f"{model_name}-encodings.th"))

    if force or not os.path.exists(os.path.join(output_save_dir, f"{model_name}-encodings-dev.th")):
        if len(dev_corpus) > 0:
            prepared_dev_corpus = create_multiple_processes_with_progress(_dist_create_tokenizer_and_process_corpus, 8, dev_corpus, model_name)
            torch.save(prepared_dev_corpus, os.path.join(output_save_dir, f"{model_name}-encodings-dev.th"))
        else:
            prepared_dev_corpus = None
    else:
        prepared_dev_corpus = torch.load(os.path.join(output_save_dir, f"{model_name}-encodings-dev.th"))

    print("collect mask prediction results for keywords")
    if force or not os.path.exists(os.path.join(output_save_dir, f"keyword_mask_prediction.json")):
        occurence_keyword = {tuple(vv):k for k,v in keyword_occurences.items() for vv in v}
        occurences = [vv for k, v in keyword_occurences.items() for vv in v]
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        prediction = _mask_prediction(0, prepared_corpus, occurences, model_name, tokenizer=tokenizer)
        keyword_mask_prediction = {k: [] for k in keyword_occurences}
        for occurence, words in zip(occurences, prediction):
            keyword_mask_prediction[occurence_keyword[tuple(occurence)]].append({"position": occurence, "prediction": words})
        with open(os.path.join(output_save_dir, f"keyword_mask_prediction.json"), "wt") as fp:
            json.dump(keyword_mask_prediction, fp, indent=4)
    else:
        with open(os.path.join(output_save_dir, f"keyword_mask_prediction.json"), "rt") as fp:
            keyword_mask_prediction = json.load(fp)
    if force or not os.path.exists(os.path.join(output_save_dir, f"keyword_occurence_filtered.json")):
        print("filtering keyword occurences")
        keyword_occurence_prediction = []
        for keyword, mask_prediction in keyword_mask_prediction.items():
            for occurence_prediction in mask_prediction:
                occurence = occurence_prediction["position"]
                prediction = occurence_prediction["prediction"]
                keyword_occurence_prediction.append((keyword, occurence, prediction))

        keyword_threshold = {
            keyword: 0 for keyword in keyword_labels
        }

        for label in label_keywords:
            if ":" in label:
                upper_label, lower_label = label.lower().split(':')
                lower_labels = lower_label.split("-")
                for l in [upper_label] + lower_labels:
                    if l not in keyword_labels:
                        keyword_labels[l] = label
            elif label not in keyword_labels:
                keyword_labels[label] = label

        filtered_occurences = create_multiple_processes_with_progress(_dist_filter_occurences, 8, keyword_occurence_prediction, keyword_labels, keyword_threshold)
        filtered_occurences.sort()
        occurence_data = []
        current_doc_id = None
        for occ, keyword in filtered_occurences:
            if len(occ) == 2:
                (doc_id, offset) = occ
                offset_ = offset + 1
            else:
                doc_id = occ[0]
                offset = occ[1]
                offset_ = offset + 2
            if current_doc_id is None or doc_id != current_doc_id:
                occurence_data.append({"tokens": corpus[int(doc_id)][1], "annotations": [[offset, offset_, keyword]], "sentence_id": int(doc_id)})
                current_doc_id = doc_id
            else:
                occurence_data[-1]['annotations'].append([offset, offset_, keyword])

        with open(os.path.join(output_save_dir, f"keyword_occurence_filtered.json"), "wt") as fp:
            json.dump(occurence_data, fp, indent=4)
    else:
        with open(os.path.join(output_save_dir, f"keyword_occurence_filtered.json"), "rt") as fp:
            occurence_data = json.load(fp)


    print("collect keyword vectors")
    if force or not os.path.exists(os.path.join(output_save_dir, f"keyword_vector.th")):
        input_data = [(prepared_corpus[t['sentence_id']], t['annotations']) for t in occurence_data]
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            model = AutoModel.from_pretrained(model_name)

        keyword_vectors = [_keyword_representation(0, input_data, tokenizer=tokenizer, model=model, model_name=model_name)]

        keyword_vectors_dict = keyword_vectors[0]
        for keyword_vector in keyword_vectors[1:]:
            for keyword in keyword_vector:
                if keyword not in keyword_vectors_dict:
                    keyword_vectors_dict[keyword] = keyword_vector[keyword]
                else:
                    keyword_vectors_dict[keyword] = (
                        keyword_vector[keyword][0] + keyword_vectors_dict[keyword][0],
                        (keyword_vector[keyword][0] * keyword_vector[keyword][1] + keyword_vectors_dict[keyword][0] * keyword_vectors_dict[keyword][1]) / (keyword_vector[keyword][0] + keyword_vectors_dict[keyword][0])
                    )

        keywordid2label = {0: "NA"}
        keyword2id = {"NA": 0}
        hidden_dim = len(list(keyword_vectors_dict.values())[0][1])
        keyword_vectors = torch.zeros(len(keyword_vectors_dict), hidden_dim, requires_grad=False)
        label_vectors = torch.zeros(len(label2id) - 1, hidden_dim, requires_grad=False)
        label_count = torch.zeros(len(label2id) - 1, requires_grad=False)
        kid = 1
        for keyword, vector in keyword_vectors_dict.items():
            keywordid2label[kid] = keyword_labels[keyword]
            lid = label2id[keyword_labels[keyword]]
            keyword_vectors[kid-1, :] = torch.from_numpy(vector[1])
            label_vectors[lid-1, :] = (keyword_vectors[kid-1] * vector[0] + label_vectors[lid-1, :] * label_count[lid-1]) / (vector[0] + label_count[lid-1])
            label_count[lid-1] += vector[0]
            keyword2id[keyword] = kid
            kid += 1
        keywordid2labelid = {kid: label2id[label] for kid, label in keywordid2label.items()}
        keyword_vectors = keyword_vectors / torch.norm(keyword_vectors, dim=1, keepdim=True)
        label_vectors = label_vectors / torch.norm(label_vectors, dim=1, keepdim=True)

        keywordid2labelid_vector_exclude_na = np.zeros((keyword_vectors.shape[0],), dtype=np.int64)
        for kid, lid in keywordid2labelid.items():
            keywordid2labelid_vector_exclude_na[kid - 1] = lid -1
        label_clusters = cluster_keyword_vectors(keyword_vectors.numpy(), label_vectors.numpy(), keywordid2labelid_vector_exclude_na)
        cluster_vectors = torch.zeros(sum(t['ncluster'] for t in label_clusters.values()), hidden_dim)
        clusterid2labelid = {0: 0}
        for labelid, clusters in label_clusters.items():
            start_index = len(clusterid2labelid) -1
            end_index = start_index+clusters["ncluster"]
            print(labelid, clusters['ncluster'], clusters['cosine'])
            cluster_vectors[start_index:end_index] = torch.from_numpy(clusters["center"])
            for cid in range(start_index + 1, end_index + 1):
                clusterid2labelid[cid] = labelid + 1
        clusterid2label = {cid: id2label[lid] for cid, lid in clusterid2labelid.items()}
        print(clusterid2labelid)
        torch.save({
            "vector": keyword_vectors,
            "keyword2id": keyword2id,
            "id2label": keywordid2label
        }, os.path.join(output_save_dir, f"keyword_vector.th"))
        torch.save({
            "vector": cluster_vectors,
            "id2label": clusterid2label
        }, os.path.join(output_save_dir, f"cluster_vector.th"))
        torch.save(label_vectors, os.path.join(output_save_dir, f"label_vector.th"))
    else:
        keyword_vector_dump = torch.load(os.path.join(output_save_dir, f"keyword_vector.th"))
        keyword_vectors = keyword_vector_dump["vector"]
        keywordid2label = keyword_vector_dump["id2label"]
        keywordid2labelid = {kid:label2id[label] for kid, label in keywordid2label.items()}
        cluster_vector_dump = torch.load(os.path.join(output_save_dir, f"cluster_vector.th"))
        cluster_vectors = cluster_vector_dump["vector"]
        clusterid2label = cluster_vector_dump["id2label"]
        clusterid2labelid = {cid: label2id[label] for cid, label in clusterid2label.items()}
        label_vectors = torch.load(os.path.join(output_save_dir, f"label_vector.th"))

    if not no_evaluate or threshold < 0:
        print("rough evaluation")
        path = example_json
        _, dataloader = get_dataset_and_loader(path, label2id=label2id, method="token", model_name=model_name)
        model = AutoModel.from_pretrained(model_name)
        best_th = {"kw": {}, "cluster": {}, "label": {}}
        best_val = {"kw": None, "cluster": None, "label": None}
        best_th.pop("kw"); best_th.pop("label")
        best_val.pop("kw"); best_val.pop("label")
        for th in range(60, 100, 2):
            current_th = th / 100
            res = {
                "kw":evaluate_vectors_on_corpus(current_th, keyword_vectors, label2id, path, model_name, "token", keywordid2labelid, model, dataloader),
                "cluster": evaluate_vectors_on_corpus(current_th, cluster_vectors, label2id, path, model_name, "token", clusterid2labelid, model, dataloader),
                "label":evaluate_vectors_on_corpus(current_th, label_vectors, label2id, path, model_name, "token", model=model, dataloader=dataloader)
                }
            for method in best_th:
                if best_val[method] is None:
                    best_val[method] = res[method]["others"].copy()
                    best_th[method] = {k:current_th for k in res[method]["others"]}
                else:
                    for event_type, best_type_result in best_val[method].items():
                        type_result = res[method]['others'][event_type]
                        f1 = type_result[2] * 2 / max(1, type_result[2] + type_result[1])
                        best_f1 = best_type_result[2] * 2 / max(1, best_type_result[2] + best_type_result[1])
                        if f1 > best_f1:
                            best_val[method][event_type] = type_result
                            best_th[method][event_type] = current_th
        overall_best = {
            method: [
                sum([t[0] for t in best_val[method].values()]),
                sum([t[1] for t in best_val[method].values()]),
                sum([t[2] for t in best_val[method].values()])
            ] for method in best_val
        }
        overall_best = {
            method: {
                'precision': v[2] / max(1, v[1]),
                'recall': v[2] / max(1, v[0]),
                'f1': v[2] * 2 / max(1, v[0] + v[1])
            } for method, v in overall_best.items()
        }
        best_val = {
            method: {
                type: {
                'precision': score[2] / max(1, score[1]),
                'recall': score[2] / max(1, score[0]),
                'f1': score[2] * 2 / max(1, score[0] + score[1])
                } for type, score in type_score.items()
            } for method, type_score in best_val.items() 
        }
        print(best_th, best_val, overall_best)
    else:
        best_th = {"kw": threshold, "cluster": threshold, "label": threshold}

    #     print("rough evaluation")
    #     if args.evaluation_json == "":
    #         path = args.example_json
    #     else:
    #         path = args.evaluation_json
    #     _, dataloader = get_dataset_and_loader(path, label2id=label2id, method="token", model_name=args.model_name)
    #     model = AutoModel.from_pretrained(args.model_name)
    #     best_th = {"kw": -1, "cluster": -1, "label": -1}
    #     best_val = {"kw": None, "cluster": None, "label": None}
    #     for th in range(60, 80, 2):
    #         current_th = th / 100
    #         res = {
    #             "kw":evaluate_vectors_on_corpus(current_th, keyword_vectors, label2id, path, args.model_name, "token", keywordid2labelid, model, dataloader),
    #             "cluster": evaluate_vectors_on_corpus(current_th, cluster_vectors, label2id, path, args.model_name, "token", clusterid2labelid, model, dataloader),
    #             "label":evaluate_vectors_on_corpus(current_th, label_vectors, label2id, path, args.model_name, "token", model=model, dataloader=dataloader)
    #             }
    #         for method in best_th:
    #             if best_val[method] is None or best_val[method]['f1'] < res[method]['f1']:
    #                 best_val[method] = res[method].copy()
    #                 best_th[method] = current_th
    #     print(best_th, best_val)
    # else:
    #     best_th = {"kw": args.threshold, "cluster": args.threshold, "label": args.threshold}

    print("start annotating")


    # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_data_kw.jsonl")):
    #     if tokenizer is None:
    #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #     if model is None:
    #         model = AutoModel.from_pretrained(args.model_name)
    #     annotations = keyword_matching(0, prepared_corpus, keyword_vectors, keywordid2label, label2id, data_sentences=data_sentences, th=best_th['kw'], uth=best_th['kw'], model_name=args.model_name, model=model, tokenizer=tokenizer)

    #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_data_kw.jsonl"), "wt") as fp:
    #         for annotation in tqdm(annotations):
    #             fp.write(json.dumps(annotation)+"\n")

    if force or not os.path.exists(os.path.join(output_save_dir, f"weakly_supervised_data_cluster.jsonl")):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            model = AutoModel.from_pretrained(model_name)
        data_sentences = _preprocess_func(train_file)
        annotations = keyword_matching(0, prepared_corpus, cluster_vectors, clusterid2label, label2id, data_sentences=data_sentences, th=best_th['cluster'], uth=best_th['cluster'], model_name=model_name, model=model, tokenizer=tokenizer)

        with open(os.path.join(output_save_dir, f"weakly_supervised_data_cluster.jsonl"), "wt") as fp:
            for annotation in tqdm(annotations):
                fp.write(json.dumps(annotation)+"\n")

    # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_data_label.jsonl")):
    #     if tokenizer is None:
    #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #     if model is None:
    #         model = AutoModel.from_pretrained(args.model_name)
    #     data_sentences = _preprocess_func(args.train_file)
    #     annotations = keyword_matching(0, prepared_corpus, label_vectors, None, label2id, data_sentences=data_sentences, th=best_th['label'], uth=best_th['label'], model_name=args.model_name, model=model, tokenizer=tokenizer)
    #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_data_label.jsonl"), "wt") as fp:
    #         for annotation in tqdm(annotations):
    #             fp.write(json.dumps(annotation)+"\n")

    if prepared_dev_corpus is not None:

        # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_dev_kw.jsonl")):
        #     if tokenizer is None:
        #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        #     if model is None:
        #         model = AutoModel.from_pretrained(args.model_name)
        #     annotations = keyword_matching(0, prepared_dev_corpus, keyword_vectors, keywordid2label, label2id, data_sentences=dev_data_sentences, th=best_th['kw'], uth=best_th['kw'], model_name=args.model_name, model=model, tokenizer=tokenizer)

        #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_dev_kw.jsonl"), "wt") as fp:
        #         for annotation in tqdm(annotations):
        #             fp.write(json.dumps(annotation)+"\n")

        if force or not os.path.exists(os.path.join(output_save_dir, f"weakly_supervised_dev_cluster.jsonl")):
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            if model is None:
                model = AutoModel.from_pretrained(model_name)
            annotations = keyword_matching(0, prepared_dev_corpus, cluster_vectors, clusterid2label, label2id, data_sentences=dev_data_sentences, th=best_th['cluster'], uth=best_th['cluster'], model_name=model_name, model=model, tokenizer=tokenizer)

            with open(os.path.join(output_save_dir, f"weakly_supervised_dev_cluster.jsonl"), "wt") as fp:
                for annotation in tqdm(annotations):
                    fp.write(json.dumps(annotation)+"\n")

        # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_dev_label.jsonl")):
        #     if tokenizer is None:
        #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        #     if model is None:
        #         model = AutoModel.from_pretrained(args.model_name)
        #     annotations = keyword_matching(0, prepared_dev_corpus, label_vectors, None, label2id, data_sentences=dev_data_sentences, th=best_th['label'], uth=best_th['label'], model_name=args.model_name, model=model, tokenizer=tokenizer)
        #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_dev_label.jsonl"), "wt") as fp:
        #         for annotation in tqdm(annotations):
        #             fp.write(json.dumps(annotation)+"\n")

def main():
    args = parse_arguments_for_weak_supervision()
    model = None
    tokenizer = None
    print(args)

    if (args.example_json == "" or args.no_evaluate) and args.threshold <= 0:
        raise ValueError("Need to provide either example data file to determine threshold value or threshold value")

    print("tokenizing and lemmatizing corpus")
    if args.force or not os.path.exists(args.corpus_jsonl):
        if args.preprocess_func == "":
            _pfunc = _preprocess_func
        else:
            nested = args.preprocess_func.split(".")
            _pmodule = ".".join(nested[:-1]); _pfunc = nested[-1]
            _pfunc = getattr(importlib.import_module(_pmodule), _pfunc)
        data_sentences = _pfunc(args.train_file)
        corpus = create_multiple_processes_with_progress(_dist_tokenize_and_lemmatize_corpus(), 8, data_sentences)
        for data in corpus: data["split"] = "train"
        if args.dev_file != "":
            dev_data_sentences = _pfunc(args.dev_file)
            dev_corpus = create_multiple_processes_with_progress(_dist_tokenize_and_lemmatize_corpus(), 8, dev_data_sentences)
            for data in dev_corpus: data["split"] = "dev"
        else:
            dev_data_sentences = []
            dev_corpus = []
        with open(args.corpus_jsonl, "wt") as fp:
            for data in corpus + dev_corpus:
                fp.write(json.dumps(data)+"\n")
        corpus = [(f"{idx}", data["tokens"], data["pos_tags"], data["lemmas"]) for idx, data in enumerate(corpus)]
        dev_corpus = [(f"{idx}", data["tokens"], data["pos_tags"], data["lemmas"]) for idx, data in enumerate(dev_corpus)]
    else:
        with open(args.corpus_jsonl, "rt") as fp:
            corpus = []
            data_sentences = []
            dev_corpus = []
            dev_data_sentences = []
            for idx, line in enumerate(fp):
                data = json.loads(line)
                if 'sentence' not in data:
                    data['sentence'] = " ".join(data['tokens'])
                if data["split"] == "train":
                    corpus.append((f"{len(corpus)}", data["tokens"], data["pos_tags"], data["lemmas"]))
                    data_sentences.append((f"{len(data_sentences)}", data['sentence']))
                else:
                    dev_corpus.append((f"{len(dev_corpus)}", data["tokens"], data["pos_tags"], data["lemmas"]))
                    dev_data_sentences.append((f"{len(dev_data_sentences)}", data['sentence']))

    print("searching keyword in corpus")
    with open(args.label_json, "rt") as fp:
        label_info = json.load(fp)
        label2id = {label: info["id"] for label, info in label_info.items()}; label2id["NA"] = 0
        id2label = {v:k for k,v in label2id.items()}
        label_keywords = _clean_keywords2({label:list(set(info["seed_keywords"])) for label, info in label_info.items()})
        keyword_labels = {vv:k for k,v in label_keywords.items() for vv in v}
        keywords = [keyword for keywords in label_keywords.values() for keyword in keywords]
    if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"keyword_occurence.json")):
        keyword_occurences = _find_keyword_occurences_in_corpus(corpus, keywords)
        with open(os.path.join(args.output_save_dir, f"keyword_occurence.json"), "wt") as fp:
            json.dump(keyword_occurences, fp, indent=4)
    else:
        with open(os.path.join(args.output_save_dir, f"keyword_occurence.json"), "rt") as fp:
            keyword_occurences = json.load(fp)

    print("prepare corpus for bert")
    if args.force or not os.path.exists(os.path.join(args.encoding_save_dir, f"{args.model_name}-encodings.th")):
        prepared_corpus = create_multiple_processes_with_progress(_dist_create_tokenizer_and_process_corpus, 8, corpus, args.model_name)
        torch.save(prepared_corpus, os.path.join(args.encoding_save_dir, f"{args.model_name}-encodings.th"))
    else:
        prepared_corpus = torch.load(os.path.join(args.encoding_save_dir, f"{args.model_name}-encodings.th"))

    if args.force or not os.path.exists(os.path.join(args.encoding_save_dir, f"{args.model_name}-encodings-dev.th")):
        if len(dev_corpus) > 0:
            prepared_dev_corpus = create_multiple_processes_with_progress(_dist_create_tokenizer_and_process_corpus, 8, dev_corpus, args.model_name)
            torch.save(prepared_dev_corpus, os.path.join(args.encoding_save_dir, f"{args.model_name}-encodings-dev.th"))
        else:
            prepared_dev_corpus = None
    else:
        prepared_dev_corpus = torch.load(os.path.join(args.encoding_save_dir, f"{args.model_name}-encodings-dev.th"))

    print("collect mask prediction results for keywords")
    if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"keyword_mask_prediction.json")):
        occurence_keyword = {tuple(vv):k for k,v in keyword_occurences.items() for vv in v}
        occurences = [vv for k, v in keyword_occurences.items() for vv in v]
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        prediction = _mask_prediction(0, prepared_corpus, occurences, args.model_name, tokenizer=tokenizer)
        keyword_mask_prediction = {k: [] for k in keyword_occurences}
        for occurence, words in zip(occurences, prediction):
            keyword_mask_prediction[occurence_keyword[tuple(occurence)]].append({"position": occurence, "prediction": words})
        with open(os.path.join(args.output_save_dir, f"keyword_mask_prediction.json"), "wt") as fp:
            json.dump(keyword_mask_prediction, fp, indent=4)
    else:
        with open(os.path.join(args.output_save_dir, f"keyword_mask_prediction.json"), "rt") as fp:
            keyword_mask_prediction = json.load(fp)
    if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"keyword_occurence_filtered.json")):
        print("filtering keyword occurences")
        keyword_occurence_prediction = []
        for keyword, mask_prediction in keyword_mask_prediction.items():
            for occurence_prediction in mask_prediction:
                occurence = occurence_prediction["position"]
                prediction = occurence_prediction["prediction"]
                keyword_occurence_prediction.append((keyword, occurence, prediction))

        keyword_threshold = {
            keyword: 0 for keyword in keyword_labels
        }

        for label in label_keywords:
            if ":" in label:
                upper_label, lower_label = label.lower().split(':')
                lower_labels = lower_label.split("-")
                for l in [upper_label] + lower_labels:
                    if l not in keyword_labels:
                        keyword_labels[l] = label
            elif label not in keyword_labels:
                keyword_labels[label] = label

        filtered_occurences = create_multiple_processes_with_progress(_dist_filter_occurences, 8, keyword_occurence_prediction, keyword_labels, keyword_threshold)
        filtered_occurences.sort()
        occurence_data = []
        current_doc_id = None
        for occ, keyword in filtered_occurences:
            if len(occ) == 2:
                (doc_id, offset) = occ
                offset_ = offset + 1
            else:
                doc_id = occ[0]
                offset = occ[1]
                offset_ = offset + 2
            if current_doc_id is None or doc_id != current_doc_id:
                occurence_data.append({"tokens": corpus[int(doc_id)][1], "annotations": [[offset, offset_, keyword]], "sentence_id": int(doc_id)})
                current_doc_id = doc_id
            else:
                occurence_data[-1]['annotations'].append([offset, offset_, keyword])

        with open(os.path.join(args.output_save_dir, f"keyword_occurence_filtered.json"), "wt") as fp:
            json.dump(occurence_data, fp, indent=4)
    else:
        with open(os.path.join(args.output_save_dir, f"keyword_occurence_filtered.json"), "rt") as fp:
            occurence_data = json.load(fp)


    print("collect keyword vectors")
    if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"keyword_vector.th")):
        input_data = [(prepared_corpus[t['sentence_id']], t['annotations']) for t in occurence_data]
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if model is None:
            model = AutoModel.from_pretrained(args.model_name)

        keyword_vectors = [_keyword_representation(0, input_data, tokenizer=tokenizer, model=model, model_name=args.model_name)]

        keyword_vectors_dict = keyword_vectors[0]
        for keyword_vector in keyword_vectors[1:]:
            for keyword in keyword_vector:
                if keyword not in keyword_vectors_dict:
                    keyword_vectors_dict[keyword] = keyword_vector[keyword]
                else:
                    keyword_vectors_dict[keyword] = (
                        keyword_vector[keyword][0] + keyword_vectors_dict[keyword][0],
                        (keyword_vector[keyword][0] * keyword_vector[keyword][1] + keyword_vectors_dict[keyword][0] * keyword_vectors_dict[keyword][1]) / (keyword_vector[keyword][0] + keyword_vectors_dict[keyword][0])
                    )

        keywordid2label = {0: "NA"}
        keyword2id = {"NA": 0}
        hidden_dim = len(list(keyword_vectors_dict.values())[0][1])
        keyword_vectors = torch.zeros(len(keyword_vectors_dict), hidden_dim, requires_grad=False)
        label_vectors = torch.zeros(len(label2id) - 1, hidden_dim, requires_grad=False)
        label_count = torch.zeros(len(label2id) - 1, requires_grad=False)
        kid = 1
        for keyword, vector in keyword_vectors_dict.items():
            keywordid2label[kid] = keyword_labels[keyword]
            lid = label2id[keyword_labels[keyword]]
            keyword_vectors[kid-1, :] = torch.from_numpy(vector[1])
            label_vectors[lid-1, :] = (keyword_vectors[kid-1] * vector[0] + label_vectors[lid-1, :] * label_count[lid-1]) / (vector[0] + label_count[lid-1])
            label_count[lid-1] += vector[0]
            keyword2id[keyword] = kid
            kid += 1
        keywordid2labelid = {kid: label2id[label] for kid, label in keywordid2label.items()}
        keyword_vectors = keyword_vectors / torch.norm(keyword_vectors, dim=1, keepdim=True)
        label_vectors = label_vectors / torch.norm(label_vectors, dim=1, keepdim=True)

        keywordid2labelid_vector_exclude_na = np.zeros((keyword_vectors.shape[0],), dtype=np.int64)
        for kid, lid in keywordid2labelid.items():
            keywordid2labelid_vector_exclude_na[kid - 1] = lid -1
        label_clusters = cluster_keyword_vectors(keyword_vectors.numpy(), label_vectors.numpy(), keywordid2labelid_vector_exclude_na)
        cluster_vectors = torch.zeros(sum(t['ncluster'] for t in label_clusters.values()), hidden_dim)
        clusterid2labelid = {0: 0}
        for labelid, clusters in label_clusters.items():
            start_index = len(clusterid2labelid) -1
            end_index = start_index+clusters["ncluster"]
            print(labelid, clusters['ncluster'], clusters['cosine'])
            cluster_vectors[start_index:end_index] = torch.from_numpy(clusters["center"])
            for cid in range(start_index + 1, end_index + 1):
                clusterid2labelid[cid] = labelid + 1
        clusterid2label = {cid: id2label[lid] for cid, lid in clusterid2labelid.items()}
        print(clusterid2labelid)
        torch.save({
            "vector": keyword_vectors,
            "keyword2id": keyword2id,
            "id2label": keywordid2label
        }, os.path.join(args.output_save_dir, f"keyword_vector.th"))
        torch.save({
            "vector": cluster_vectors,
            "id2label": clusterid2label
        }, os.path.join(args.output_save_dir, f"cluster_vector.th"))
        torch.save(label_vectors, os.path.join(args.output_save_dir, f"label_vector.th"))
    else:
        keyword_vector_dump = torch.load(os.path.join(args.output_save_dir, f"keyword_vector.th"))
        keyword_vectors = keyword_vector_dump["vector"]
        keywordid2label = keyword_vector_dump["id2label"]
        keywordid2labelid = {kid:label2id[label] for kid, label in keywordid2label.items()}
        cluster_vector_dump = torch.load(os.path.join(args.output_save_dir, f"cluster_vector.th"))
        cluster_vectors = cluster_vector_dump["vector"]
        clusterid2label = cluster_vector_dump["id2label"]
        clusterid2labelid = {cid: label2id[label] for cid, label in clusterid2label.items()}
        label_vectors = torch.load(os.path.join(args.output_save_dir, f"label_vector.th"))

    if not args.no_evaluate or args.threshold < 0:
        print("rough evaluation")
        if args.evaluation_json == "":
            path = args.example_json
        else:
            path = args.evaluation_json
        _, dataloader = get_dataset_and_loader(path, label2id=label2id, method="token", model_name=args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        best_th = {"kw": {}, "cluster": {}, "label": {}}
        best_val = {"kw": None, "cluster": None, "label": None}
        best_th.pop("kw"); best_th.pop("label")
        best_val.pop("kw"); best_val.pop("label")
        for th in range(60, 100, 2):
            current_th = th / 100
            res = {
                "kw":evaluate_vectors_on_corpus(current_th, keyword_vectors, label2id, path, args.model_name, "token", keywordid2labelid, model, dataloader),
                "cluster": evaluate_vectors_on_corpus(current_th, cluster_vectors, label2id, path, args.model_name, "token", clusterid2labelid, model, dataloader),
                "label":evaluate_vectors_on_corpus(current_th, label_vectors, label2id, path, args.model_name, "token", model=model, dataloader=dataloader)
                }
            for method in best_th:
                if best_val[method] is None:
                    best_val[method] = res[method]["others"].copy()
                    best_th[method] = {k:current_th for k in res[method]["others"]}
                else:
                    for event_type, best_type_result in best_val[method].items():
                        type_result = res[method]['others'][event_type]
                        f1 = type_result[2] * 2 / max(1, type_result[2] + type_result[1])
                        best_f1 = best_type_result[2] * 2 / max(1, best_type_result[2] + best_type_result[1])
                        if f1 > best_f1:
                            best_val[method][event_type] = type_result
                            best_th[method][event_type] = current_th
        overall_best = {
            method: [
                sum([t[0] for t in best_val[method].values()]),
                sum([t[1] for t in best_val[method].values()]),
                sum([t[2] for t in best_val[method].values()])
            ] for method in best_val
        }
        overall_best = {
            method: {
                'precision': v[2] / max(1, v[1]),
                'recall': v[2] / max(1, v[0]),
                'f1': v[2] * 2 / max(1, v[0] + v[1])
            } for method, v in overall_best.items()
        }
        best_val = {
            method: {
                type: {
                'precision': score[2] / max(1, score[1]),
                'recall': score[2] / max(1, score[0]),
                'f1': score[2] * 2 / max(1, score[0] + score[1])
                } for type, score in type_score.items()
            } for method, type_score in best_val.items() 
        }
        print(best_th, best_val, overall_best)
    else:
        best_th = {"kw": args.threshold, "cluster": args.threshold, "label": args.threshold}

    #     print("rough evaluation")
    #     if args.evaluation_json == "":
    #         path = args.example_json
    #     else:
    #         path = args.evaluation_json
    #     _, dataloader = get_dataset_and_loader(path, label2id=label2id, method="token", model_name=args.model_name)
    #     model = AutoModel.from_pretrained(args.model_name)
    #     best_th = {"kw": -1, "cluster": -1, "label": -1}
    #     best_val = {"kw": None, "cluster": None, "label": None}
    #     for th in range(60, 80, 2):
    #         current_th = th / 100
    #         res = {
    #             "kw":evaluate_vectors_on_corpus(current_th, keyword_vectors, label2id, path, args.model_name, "token", keywordid2labelid, model, dataloader),
    #             "cluster": evaluate_vectors_on_corpus(current_th, cluster_vectors, label2id, path, args.model_name, "token", clusterid2labelid, model, dataloader),
    #             "label":evaluate_vectors_on_corpus(current_th, label_vectors, label2id, path, args.model_name, "token", model=model, dataloader=dataloader)
    #             }
    #         for method in best_th:
    #             if best_val[method] is None or best_val[method]['f1'] < res[method]['f1']:
    #                 best_val[method] = res[method].copy()
    #                 best_th[method] = current_th
    #     print(best_th, best_val)
    # else:
    #     best_th = {"kw": args.threshold, "cluster": args.threshold, "label": args.threshold}

    print("start annotating")


    # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_data_kw.jsonl")):
    #     if tokenizer is None:
    #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #     if model is None:
    #         model = AutoModel.from_pretrained(args.model_name)
    #     annotations = keyword_matching(0, prepared_corpus, keyword_vectors, keywordid2label, label2id, data_sentences=data_sentences, th=best_th['kw'], uth=best_th['kw'], model_name=args.model_name, model=model, tokenizer=tokenizer)

    #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_data_kw.jsonl"), "wt") as fp:
    #         for annotation in tqdm(annotations):
    #             fp.write(json.dumps(annotation)+"\n")

    if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_data_cluster.jsonl")):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if model is None:
            model = AutoModel.from_pretrained(args.model_name)
        data_sentences = _preprocess_func(args.train_file)
        annotations = keyword_matching(0, prepared_corpus, cluster_vectors, clusterid2label, label2id, data_sentences=data_sentences, th=best_th['cluster'], uth=best_th['cluster'], model_name=args.model_name, model=model, tokenizer=tokenizer)

        with open(os.path.join(args.output_save_dir, f"weakly_supervised_data_cluster.jsonl"), "wt") as fp:
            for annotation in tqdm(annotations):
                fp.write(json.dumps(annotation)+"\n")

    # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_data_label.jsonl")):
    #     if tokenizer is None:
    #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #     if model is None:
    #         model = AutoModel.from_pretrained(args.model_name)
    #     data_sentences = _preprocess_func(args.train_file)
    #     annotations = keyword_matching(0, prepared_corpus, label_vectors, None, label2id, data_sentences=data_sentences, th=best_th['label'], uth=best_th['label'], model_name=args.model_name, model=model, tokenizer=tokenizer)
    #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_data_label.jsonl"), "wt") as fp:
    #         for annotation in tqdm(annotations):
    #             fp.write(json.dumps(annotation)+"\n")

    if prepared_dev_corpus is not None:

        # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_dev_kw.jsonl")):
        #     if tokenizer is None:
        #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        #     if model is None:
        #         model = AutoModel.from_pretrained(args.model_name)
        #     annotations = keyword_matching(0, prepared_dev_corpus, keyword_vectors, keywordid2label, label2id, data_sentences=dev_data_sentences, th=best_th['kw'], uth=best_th['kw'], model_name=args.model_name, model=model, tokenizer=tokenizer)

        #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_dev_kw.jsonl"), "wt") as fp:
        #         for annotation in tqdm(annotations):
        #             fp.write(json.dumps(annotation)+"\n")

        if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_dev_cluster.jsonl")):
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            if model is None:
                model = AutoModel.from_pretrained(args.model_name)
            annotations = keyword_matching(0, prepared_dev_corpus, cluster_vectors, clusterid2label, label2id, data_sentences=dev_data_sentences, th=best_th['cluster'], uth=best_th['cluster'], model_name=args.model_name, model=model, tokenizer=tokenizer)

            with open(os.path.join(args.output_save_dir, f"weakly_supervised_dev_cluster.jsonl"), "wt") as fp:
                for annotation in tqdm(annotations):
                    fp.write(json.dumps(annotation)+"\n")

        # if args.force or not os.path.exists(os.path.join(args.output_save_dir, f"weakly_supervised_dev_label.jsonl")):
        #     if tokenizer is None:
        #         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        #     if model is None:
        #         model = AutoModel.from_pretrained(args.model_name)
        #     annotations = keyword_matching(0, prepared_dev_corpus, label_vectors, None, label2id, data_sentences=dev_data_sentences, th=best_th['label'], uth=best_th['label'], model_name=args.model_name, model=model, tokenizer=tokenizer)
        #     with open(os.path.join(args.output_save_dir, f"weakly_supervised_dev_label.jsonl"), "wt") as fp:
        #         for annotation in tqdm(annotations):
        #             fp.write(json.dumps(annotation)+"\n")


if __name__ == "__main__":
    main()
