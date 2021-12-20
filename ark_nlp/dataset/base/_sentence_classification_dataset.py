"""
# Copyright Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import copy
import torch
import pandas as pd

from functools import lru_cache
from torch.utils.data import Dataset
from ark_nlp.dataset.base._dataset import BaseDataset


class SentenceClassificationDataset(BaseDataset):
        
    def _get_categories(self):
        return sorted(list(set([data['label'] for data in self.dataset])))
    
    def _convert_to_dataset(self, data_df):
        
        dataset = []
        
        data_df['text'] = data_df['text'].apply(lambda x: x.lower().strip())
        
        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({feature_name_: getattr(row_, feature_name_) 
                             for feature_name_ in feature_names})
            
        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text'])              
            
            input_ids, input_mask, segment_ids = input_ids

            feature = {
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids
                        
            features.append(feature)
        
        return features        

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = vanilla_tokenizer.tokenize(row_['text'])
            length = len(tokens)
            input_ids = vanilla_tokenizer.sequence_to_ids(tokens)  

            feature = {
                'input_ids': input_ids,
                'length': length if length < vanilla_tokenizer.max_seq_len else vanilla_tokenizer.max_seq_len
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids
            
            features.append(feature)
        
        return features


class PairSentenceClassificationDataset(BaseDataset):
        
    def _get_categories(self):
        return sorted(list(set([data['label'] for data in self.dataset])))
    
    def _convert_to_dataset(self, data_df):
        
        dataset = []
        
        data_df['text_a'] = data_df['text_a'].apply(lambda x: x.lower().strip())
        data_df['text_b'] = data_df['text_b'].apply(lambda x: x.lower().strip())
        
        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({feature_name_: getattr(row_, feature_name_) 
                             for feature_name_ in feature_names})
            
        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text_a'], row_['text_b'])
            
            input_ids, input_mask, segment_ids = input_ids
                        
            input_a_length = self._get_input_length(row_['text_a'], bert_tokenizer)
            input_b_length = self._get_input_length(row_['text_b'], bert_tokenizer)

            feature = {
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)
        
        return features        

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):

            input_a_ids = vanilla_tokenizer.sequence_to_ids(row_['text_a'])
            input_b_ids = vanilla_tokenizer.sequence_to_ids(row_['text_b'])   

            feature = {
                'input_a_ids': input_a_ids,
                'input_b_ids': input_b_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids
            
            features.append(feature)
        
        return features