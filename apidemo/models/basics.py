import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel
import math


class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self,
                 dimensions,
                 activation='relu',
                 dropout_prob=0.0,
                 bias=True):
        super(Linears, self).__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i],
                                               dimensions[i + 1],
                                               bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        inputs = self.dropout(inputs)
        return inputs


class BERTEncoder(nn.Module):
    def __init__(self,
                 bert_model="bert-large-cased",
                 output_layers=-1,
                 dropout=0.0,
                 cache_dir=None):
        super(BERTEncoder, self).__init__()
        bert_config = BertConfig.from_pretrained(bert_model,
                                                 cache_dir=cache_dir,
                                                 output_hidden_states=True)
        self.bert = BertModel.from_pretrained(bert_model,
                                              cache_dir=cache_dir,
                                              config=bert_config)
        self.hidden_size = bert_config.hidden_size
        self.drop = nn.Dropout(dropout)
        if not isinstance(output_layers, list):
            self.output_layers = [output_layers]
        else:
            self.output_layers = list(set([int(t) for t in output_layers]))

    def forward(self, input_ids, attention_mask):
        _, _, hidden = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask)
        output = torch.cat([hidden[layer] for layer in self.output_layers], -1)
        return self.drop(output)

    def merge_pieces(self, sequence, token_spans):
        return torch.matmul(token_spans, sequence)


class BaseTokenEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseTokenEncoder, self).__init__()
        if "input_size" in kwargs:
            self.input_size = kwargs["input_size"]
        elif len(args) > 0:
            self.input_size = args[0]
        else:
            raise TypeError("Input size should be set for a token encoder.")
        if "output_size" in kwargs:
            self.output_size = kwargs["output_size"]
        elif len(args) > 1:
            self.output_size = args[1]
        else:
            raise TypeError("Output size should be set for a token encoder.")


class CNNEncoder(BaseTokenEncoder):
    
    def __init__(self, *args, **kwargs):
        super(CNNEncoder, self).__init__(*args, **kwargs)
        if "kernel_size" in kwargs:
            self.kernel_size = kwargs["kernel_size"]
        else:
            self.kernel_size = 3
        if "dilation" in kwargs:
            self.dilation = kwargs["dilation"]
        else:
            self.dilation = 1
        if "activation" in kwargs:
            self.activation = kwargs["activation"]
        else:
            self.activation = nn.ReLU()
        # context conv
        self.padding = self.dilation * (self.kernel_size // 2)
        self.truncate = (1 + self.kernel_size) % 2
        # temporal conv
        # self.padding = self.dilation * (self.kernel_size - 1)
        self.encoder = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.output_size,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            dilation=self.dilation
            )

    def forward(self, sentence, mask=None): 
        sentence_length = sentence.size(1)
        sentence = self.activation(self.encoder(sentence.transpose(1, 2))).transpose(1, 2)
        if mask is None:
            sentence_pooled = F.max_pool1d(sentence.transpose(1, 2), sentence.size(1)).squeeze(2)
        else:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.float()
            sentence = sentence[:, :sentence_length, :]
            sentence = sentence * mask
            sentence_pooled = F.max_pool1d((sentence + (mask - 1) * 100).transpose(1, 2), sentence.size(1)).squeeze(2)
        return sentence, sentence_pooled


class TransEncoder(BaseTokenEncoder):
    def __init__(self, *args, **kwargs):
        super(TransEncoder, self).__init__(*args, **kwargs)
        if self.input_size != self.output_size:
            self.resize = True
            self.resize_layer = nn.Sequential(
                nn.Linear(self.input_size, self.output_size),
                nn.ReLU()
            )
        else:
            self.resize = False
        if "hidden_size" in kwargs:
            self.hidden_size = kwargs["hidden_size"]
        else:
            self.hidden_size = 2048  
        if "activation" in kwargs:
            self.activation = kwargs["activation"]
        else:
            self.activation = 'relu'
        if "num_head" in kwargs:
            self.num_head = kwargs["num_head"]
        else:
            self.num_head = 8
        if "num_layer" in kwargs:
            self.num_layer = kwargs["num_layer"]
        else:
            self.num_layer = 2
        if "dropout" in kwargs:
            self.dropout = kwargs["dropout"]
        else:
            self.dropout = 0.1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_size, 
            nhead=self.num_head,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation
            )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=self.num_layer
            )
    
    def forward(self, sentence, mask=None, src_mask=None, cls_token_pos=0):
        sentence = sentence.transpose(0, 1)
        sentence = self.encoder(sentence, mask=src_mask, src_key_padding_mask=~mask)
        if self.resize:
            sentence = self.resize_layer(sentence)
        sentence = sentence.transpose(0, 1)
        conclusion = sentence[:, cls_token_pos, :]
        self.output = sentence
        self.conclusion = conclusion
        return sentence, conclusion