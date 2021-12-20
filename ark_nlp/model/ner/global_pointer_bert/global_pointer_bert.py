import time
import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import BertModel
from transformers import BertPreTrainedModel
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer


class GlobalPointerBert(BertForTokenClassification):
    """
    基于GlobalPointe的命名实体模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns: 

    Reference:
        [1] https://www.kexue.fm/archives/8373
        [2] https://github.com/suolyer/PyTorch_BERT_Biaffine_NER
    """ 
    def __init__(
        self, 
        config, 
        encoder_trained=True
    ):
        super(GlobalPointerBert, self).__init__(config)
        
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        
        for param in self.bert.parameters():
            param.requires_grad = encoder_trained 
        
        self.global_pointer = GlobalPointer(self.num_labels, 64, 768)

        self.init_weights()

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True, 
                            output_hidden_states=True
                           ).hidden_states
        

        sequence_output = outputs[-1]
        
        logits = self.global_pointer(sequence_output, mask=attention_mask)
        
        return logits