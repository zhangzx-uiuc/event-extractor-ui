import torch
import torch.nn as nn
import transformers
from transformers import BatchEncoding
from typing import *
import re

IO_match = re.compile(r'(?P<start>\d+)I-(?P<label>\S+)\s(?:(?P<end>\d+)I-(?P=label)\s)*')


def to_char(predictions:List[Union[List[Tuple[int, int, str]], Set[Tuple[int, int, str]]]], encodings:Union[List[BatchEncoding], BatchEncoding]) -> List[Union[List[Tuple[int, int, str]], Set[Tuple[int, int, str]]]]:
        fw = None
        corpus_annotations = []
        for i, prediction in enumerate(predictions):
            if isinstance(encodings, list):
                encoding = encodings[i]
            annotations = []
            for annotation in prediction:
                start_pt = annotation[0]
                end_pt = annotation[1]
                if isinstance(encodings, list):
                    start = encoding.token_to_chars(start_pt).start
                    end = encoding.token_to_chars(end_pt-1).end
                else:
                    start = encodings.token_to_chars(i, start_pt).start
                    end = encodings.token_to_chars(i, end_pt-1).end
                annotations.append([start, end, annotation[2]])
            corpus_annotations.append(annotations)
        return corpus_annotations

class IEToken(nn.Module):
    def __init__(self, nclass:int, model_name:str, id2label:Dict[int, str], **kwargs):
        super().__init__()
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.linear_map = nn.Linear(2048, nclass)
        self.crit = nn.CrossEntropyLoss()
        self.id2label = id2label

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

def find_offsets(seq_str:str, match:re.Pattern):
    annotations = []
    for annotation in match.finditer(seq_str):
        start = int(annotation.group('start'))
        label = annotation.group('label')
        end = annotation.group('end')
        end = start + 1 if end is None else int(end) + 1
        annotations.append((start, end, label))
    return annotations
    
def collect_spans(sequence:str, tag2label:Optional[Dict[str, str]]=None) -> Set[Tuple[int, int, str]]:
    spans = find_offsets(sequence, IO_match)
    if tag2label:
        label_spans = set()
        for span in spans:
            label_spans.add((span[0], span[1], tag2label[span[2]]))
    else:
        label_spans = set(spans)
    return label_spans
    
def annotate(sentence, model:IEToken, batch_size=8, max_length=96):
    # print(sentence)
    # print(model.id2label)
    # return [[{"trigger": [80, 86, "Transaction:Transfer-Money"], "arguments": [[20, 30, "Giver"], [87, 92, "Recipient"]]}]]
    model.eval()
    label2tag = {
        v: v.replace("-", "_") for v in model.id2label.values()
    }
    tag2label = {
        v:k for k,v in label2tag.items()
    }
    def get_tag(id):
        if id == 0:
            return 'O'
        else:
            return f'I-{label2tag[model.id2label[id]]}'
    annotations = []
    with torch.no_grad():
        for i in range(0, len(sentence), batch_size):
            encoded = model.tokenizer(
                    text=sentence[i:i+batch_size],
                    max_length=max_length,
                    is_split_into_words=isinstance(sentence[0], list),
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=False,
                    return_tensors='pt'
                )
            encoded.to(model.linear_map.weight.device)
            input_ids, attention_masks = encoded["input_ids"], encoded["attention_mask"]
            hidden = model.pretrained_lm(input_ids, attention_masks, output_hidden_states=True)
            hidden = torch.cat((hidden.last_hidden_state, hidden.hidden_states[-3]), dim=-1)
            outputs = model.linear_map(hidden)
            preds = torch.argmax(outputs, dim=-1)
            preds[attention_masks==0] = -100
            preds = preds.cpu().numpy()
            sequences = []
            for idx, sequence in enumerate(preds):
                sequence = sequence[sequence!=-100]
                sequences.append(" ".join([f'{offset}{get_tag(token)}' for offset, token in enumerate(sequence)]) + " ")
            sequences = [list(collect_spans(sequence, tag2label)) for sequence in sequences]
            annotations.extend(to_char(sequences, encoded))

    results = []
    for k,trigger in enumerate(annotations[0]):
        result_k = {"trigger": trigger, "arguments": []}
        results.append(result_k)

    return [results]


def annotate_arguments(trigger_annotations, sentence, arg_model, tokenizer, spacy_model, nltk_tokenizer):
    data_input = {"sentence": sentence, "events": trigger_annotations[0]}
    output_res = arg_model.predict_one_example(tokenizer, data_input, spacy_model, nltk_tokenizer)
    return [output_res["events"]]


def load_ckpt(ckpt_path:str, device="cuda:0"):
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    state_dict = ckpt['state_dict']
    nclass = state_dict['linear_map.weight'].size(0)
    id2label = ckpt['id2label']
    model = IEToken(nclass, "roberta-large", id2label)
    model.to(torch.device(device))
    model.load_state_dict(state_dict=state_dict)
    return model