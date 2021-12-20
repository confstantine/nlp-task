import torch
import numpy as np
from torch import nn
from collections import Counter
from transformers import BertPreTrainedModel, BertModel


class Bert(BertPreTrainedModel):
    """
    原始的BERT模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns:

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    """

    def __init__(
            self,
            config,
            encoder_trained=True,
            pooling='cls'
    ):
        super(Bert, self).__init__(config)

        self.bert = BertModel(config)
        self.pooling = pooling

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size + 10, self.num_labels)

        self.relative_pos_embedding = nn.Embedding(4, 10)

        self.init_weights()

    def mask_pooling(self, x, attention_mask=None):
        if attention_mask is None:
            return torch.mean(x, dim=1)
        return torch.sum(x * attention_mask.unsqueeze(2), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)

    def sequence_pooling(self, sequence_feature, attention_mask):
        if self.pooling == 'first_last_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[1]
        elif self.pooling == 'last_avg':
            sequence_feature = sequence_feature[-1]
        elif self.pooling == 'last_2_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[-2]
        elif self.pooling == 'cls':
            return sequence_feature[-1][:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(self.pooling))

        return self.mask_pooling(sequence_feature, attention_mask)

    def get_encoder_feature(self, encoder_output, attention_mask):
        if self.task == 'SequenceLevel':
            return self.sequence_pooling(encoder_output, attention_mask)
        elif self.task == 'TokenLevel':
            return encoder_output[-1]
        else:
            return encoder_output[-1][:, 0, :]

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            speaker_ids=None,
            **kwargs
    ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True,
                            output_hidden_states=True
                            ).hidden_states

        #         encoder_feature = self.get_encoder_feature(outputs, attention_mask)

        speaker_feature = self.relative_pos_embedding(speaker_ids)
        #         encoder_feature = outputs[-1] + speaker_feature

        encoder_feature = torch.cat([outputs[-1], speaker_feature], dim=-1)
        encoder_feature = self.mask_pooling(encoder_feature, attention_mask)

        encoder_feature = self.dropout(encoder_feature)
        out = self.classifier(encoder_feature)

        return out


class TMPredictor(object):
    def __init__(
            self,
            modules,
            tokernizer,
            cat2id
    ):

        self.modules = modules
        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.modules[0].parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text_a,
            text_b
    ):
        input_ids = self.tokenizer.sequence_to_ids(text_a, text_b)
        input_ids, input_mask, segment_ids, speaker_ids, e1_mask = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'speaker_ids': speaker_ids
        }
        return features

    def _convert_to_vanilla_ids(
            self,
            text_a,
            text_b
    ):
        input_ids = self.tokenizer.sequence_to_ids(text_a, text_b)

        features = {
            'input_ids': input_ids
        }
        return features

    def _get_input_ids(
            self,
            text_a,
            text_b
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(text_a, text_b)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text,
            topk=None,
            return_label_name=True,
            return_proba=False
    ):
        if topk == None:
            topk = len(self.cat2id) if len(self.cat2id) > 2 else 1
        text_a, text_b = text
        features = self._get_input_ids(text_a, text_b)
        # self.module.eval()

        preds = []
        probas = []
        vote_label_idx = []

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)

            logits = 0
            weight_sum = 0
            for idx, module in enumerate(self.modules):
                logit = self.modules[idx](**inputs) * 1
                logit = torch.nn.functional.softmax(logit, dim=1)

                probs, indices = logit.topk(topk, dim=1, sorted=True)

                preds.append(indices.cpu().numpy()[0][0])
                rank = indices.cpu().numpy()[0]
                rank_dict = {_index: prob for _index, prob in zip(rank, probs.cpu().numpy()[0])}
                probas.append([rank_dict[_index] for _index in range(len(rank))])

        most_ = Counter(preds).most_common(len(self.id2cat))
        #         print(most_)

        max_vote_num = most_[0][1]
        most_ = [m for m in most_ if m[1] != 1]  # 剔除1票的相同者
        most_ = [m for m in most_ if m[1] == max_vote_num]  # 只选择等于投票最大值的
        if len(most_) == 0:  # 如果全是1票
            vote_label_idx.append(Counter(preds).most_common(1)[0][0])
        elif len(most_) == 1:
            vote_label_idx.append(most_[0][0])
        else:
            prob_list_np = np.array(probas)
            select_rank = -10000
            select_m = 10000
            for m, num in most_:
                # 拿概率第m列（所有模型对第m列的概率）求和
                prob_m = prob_list_np[:, m]
                if sum(prob_m) > select_rank:
                    select_m = m
                    select_rank = sum(prob_m)

            vote_label_idx.append(select_m)

        if vote_label_idx[0] == -1:
            print(most_)

            print(probas)

        return self.id2cat[vote_label_idx[0]]
