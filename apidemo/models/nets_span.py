import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from typing import *

class IESPAN(nn.Module):
    def __init__(self, hidden_dim:int, nclass:int, model_name:str, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.linear_map = nn.Linear(2 * d_model, nclass)
        self.dropout = nn.Dropout(0.0)
        self.label_info = None
        self.train_labels = -1
        self.crit = nn.CrossEntropyLoss()

    def forward(self, batch):
        token_ids, attention_masks, spans, labels = batch["input_ids"], batch["attention_mask"], batch['spans'], batch["labels"]
        encoded = self.pretrained_lm(token_ids, attention_masks, output_hidden_states=True)
        encoded = torch.cat((encoded.last_hidden_state, encoded.hidden_states[-3]), dim=-1)
        candidates = torch.matmul(spans, encoded)
        span_mask = labels >= 0

        outputs = self.linear_map(candidates)
        loss = self.crit(outputs[span_mask], labels[span_mask])
        preds = torch.argmax(outputs, dim=-1)

        ngold = torch.sum((labels > 0)[span_mask].float())
        npred = torch.sum((preds > 0)[span_mask].float())
        match = torch.sum(((preds == labels)*(labels > 0))[span_mask].float())

        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach(),
            "f1": torch.tensor([ngold, npred, match])
            }

class IEToken(nn.Module):
    def __init__(self, hidden_dim:int, nclass:int, model_name:str, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.linear_map = nn.Linear(d_model*2, nclass)
        self.crit = nn.CrossEntropyLoss()
        self.outputs = dict()

    def compute_cross_entropy(self, logits, labels):
        mask = labels >= 0
        return self.crit(logits[mask], labels[mask])

    def forward(self, batch):
        token_ids, attention_masks, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        encoded = self.pretrained_lm(token_ids, attention_masks, output_hidden_states=True)
        encoded = torch.cat((encoded.last_hidden_state, encoded.hidden_states[-3]), dim=-1)
        outputs = self.linear_map(encoded)
        loss = self.compute_cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)
        preds[labels < 0] = labels[labels < 0]
        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }

class IEFromNLI(nn.Module):
    def __init__(self, model_name:str="roberta-large-mnli", distributed:bool=False, **kwargs):
        super().__init__()
        self.pretrained_lm = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model_name = model_name
        self.distributed=distributed
        if self.distributed:
            self.crit = nn.CrossEntropyLoss(reduction='none')
        else:
            self.crit = nn.CrossEntropyLoss()
        self.ref_grads = None

    def forward(self, batch, **kwargs):
        labels = batch["labels"]
        logits = self.pretrained_lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        positive_logits = logits[:, 2]
        negative_logits = logits[:, 0] + logits[:, 1]
        logits = torch.stack((negative_logits, positive_logits), dim=-1)
        loss = self.crit(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }
