import argparse
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from transformers import BertConfig, BertModel, BertTokenizer

class Model(nn.Module):
    def __init__(self, num_labels, config, regression):
        super().__init__()
        self.num_labels = num_labels
        config.num_labels = num_labels
        self.config = config
        self.bert = BertModel.from_pretrained("bert-base-uncased", config = config)
        for param in self.bert.parameters():
            param.requires_grad = False
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.Linear1 = nn.Linear(config.hidden_size, 256)
        self.Linear2 = nn.Linear(256, 64)
        self.Linear3 = nn.Linear(64,16)
        self.Linear1_dis = nn.Linear(self.config.hidden_size, 512)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(16, config.num_labels)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.tanh=  nn.Tanh()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print("return_dict", return_dict)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #print(outputs)
        pooled_output = outputs[1]
        
        temp_outputs = self.Linear1(pooled_output)
        temp_outputs = self.batchnorm1(temp_outputs)
        temp_outputs = self.relu(temp_outputs)
        
        temp_outputs = self.Linear2(temp_outputs)
        temp_outputs = self.batchnorm2(temp_outputs)
        temp_outputs = self.relu(temp_outputs)
        temp_outputs = self.dropout(temp_outputs)
        
        temp_outputs = self.Linear3(temp_outputs)
        temp_outputs = self.batchnorm3(temp_outputs)
        temp_outputs = self.relu(temp_outputs)
        
        logits = self.classifier(temp_outputs)
        if(self.config.num_labels==1) :
            return self.tanh(logits)
        return self.log_softmax(logits)

    def predict_class(self, pred: torch.Tensor) -> List:
        class_outputs = []

        for output in pred:
            output = torch.exp(output)
            class_outputs.append(output.tolist().index(max(output.tolist())))

        return class_outputs

    def predict_proba(self, pred: torch.Tensor) -> List:
        softmax_outputs = []

        for output in pred:
            softmax_outputs.append(F.softmax(output, dim=0))

        return softmax_outputs

    def weight_reset(self) -> None:
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()



def getModelAndTokenizer(path, num_labels, regression):
    tokenizer = BertTokenizer.from_pretrained(path)
    config  = BertConfig.from_pretrained(path, output_hidden_states=True, output_attentions=True)
    if regression:
        config.num_labels=1
    else:
        config.num_labels = num_labels
    model = Model(num_labels, config, regression)
    return (tokenizer, model)
