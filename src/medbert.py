import codecs
import json
import os
import sys
sys.path.append("../")
sys.path.append("../transformers/src")
import copy
import gc
import torch
import pickle
from tqdm import tqdm
from utils import set_seed, get_task_data, random_split_train_and_dev
from dataset import PairSentenceClassificationDataset
from transformers import AutoTokenizer, BertConfig
from tokenizer import TransfomerTokenizer
from sklearn.model_selection import KFold
from model import Bert, TMPredictor
from finetune import SequenceClassificationTask


class TMDataset(PairSentenceClassificationDataset):
    def __init__(self, *args, **kwargs):
        super(TMDataset, self).__init__(*args, **kwargs)
        self.categories_b = sorted(list(set([data['label_b'] for data in self.dataset])))
        self.cat2id_b = dict(zip(self.categories_b, range(len(self.categories_b))))
        self.id2cat_b = dict(zip(range(len(self.categories_b)), self.categories_b))

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        features = []
        for (index_, row_) in tqdm(enumerate(self.dataset)):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text_a'], row_['text_b'])

            input_ids, input_mask, segment_ids, speaker_ids, e1_mask = input_ids

            # input_a_length = self._get_input_length(row_['text_a'], bert_tokenizer)
            # input_b_length = self._get_input_length(row_['text_b'], bert_tokenizer)

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'speaker_ids': speaker_ids,
                'e1_mask': e1_mask
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                label_ids_b = self.cat2id_b[row_['label_b']]

                feature['label_ids'] = label_ids
                feature['label_ids_b'] = label_ids_b

            features.append(feature)

        return features


def freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


if __name__ == "__main__":
    set_seed(2021)
    model_name_or_path = "../pretrained_models/medbert"
    max_seq_length = 256

    data_df = get_task_data('../data/source_datasets/train.jsonl')
    train_data_df, dev_data_df = random_split_train_and_dev(data_df, split_rate=0.8)
    tm_train_dataset = TMDataset(train_data_df)
    tm_dev_dataset = TMDataset(dev_data_df, categories=tm_train_dataset.categories)
    bert_vocab = AutoTokenizer.from_pretrained(model_name_or_path)
    bert_vocab.add_special_tokens({'additional_special_tokens': ["[unused1]", "[unused2]", "|"]})
    tokenizer = TransfomerTokenizer(bert_vocab, max_seq_length)
    if os.path.exists("../cache/tm_dataset.pkl"):
        tm_dataset = pickle.load(open("../cache/tm_dataset.pkl", "rb"))
    else:
        tm_dataset = TMDataset(data_df)
        tm_dataset.convert_to_ids(tokenizer)
        pickle.dump(tm_dataset, open("../cache/tm_dataset.pkl", "wb"))

    kf = KFold(5, shuffle=True, random_state=42)
    examples = copy.deepcopy(tm_dataset.dataset)
    for fold_, (train_ids, dev_ids) in enumerate(kf.split(examples)):
        print(f"start fold{fold_}")
        tm_train_dataset.dataset = [examples[_idx] for _idx in train_ids]
        tm_dev_dataset.dataset = [examples[_idx] for _idx in dev_ids]

        bert_config = BertConfig.from_pretrained(model_name_or_path,
                                                 num_labels=len(tm_train_dataset.cat2id))
        bert_config.gradient_checkpointing = True
        dl_module = Bert.from_pretrained(model_name_or_path,
                                         config=bert_config)
        # freeze_params(dl_module.bert.embeddings)
        param_optimizer = list(dl_module.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        model = SequenceClassificationTask(dl_module, 'adamw', 'lsce', cuda_device=0, ema_decay=0.995)

        save_module_path = '../checkpoint/medbert2/'
        os.makedirs(save_module_path, exist_ok=True)
        model.fit(tm_train_dataset,
                  tm_dev_dataset,
                  lr=2e-5,
                  epochs=1,
                  batch_size=64,
                  params=optimizer_grouped_parameters,
                  evaluate_save=True,
                  save_module_path=save_module_path + str(fold_) + '.pth'
                  )

        del dl_module
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # predict
    ensemble_dl_modules = []
    bert_config = BertConfig.from_pretrained(model_name_or_path,
                                             num_labels=len(tm_dataset.cat2id))
    for file_name_ in os.listdir('../checkpoint/medbert2/'):
        if file_name_.startswith('.'):
            continue
        ensemble_dl_module = Bert(config=bert_config)
        ensemble_dl_module.load_state_dict(torch.load('../checkpoint/medbert2/' + file_name_))
        ensemble_dl_module.eval()
        ensemble_dl_module.to('cuda:0')
        ensemble_dl_modules.append(ensemble_dl_module)

    tm_predictor_instance = TMPredictor(ensemble_dl_modules, tokenizer, tm_dataset.cat2id)
    submit_result = []
    with codecs.open('../data/source_datasets/testa.txt', mode='r', encoding='utf8') as f:
        reader = f.readlines(f)

    data_list = []

    for dialogue_ in tqdm(reader):
        dialogue_ = json.loads(dialogue_)
        for content_idx_, contents_ in enumerate(dialogue_['dialog_info']):

            terms_ = contents_['ner']

            if len(terms_) != 0:
                idx_ = 0
                for _ner_idx, term_ in enumerate(terms_):

                    entity_ = dict()

                    entity_['dialogue'] = dialogue_

                    _text = dialogue_['dialog_info'][content_idx_]['text']
                    _text_list = list(_text)
                    _text_list.insert(term_['range'][0], '[unused1]')
                    _text_list.insert(term_['range'][1] + 1, '[unused2]')
                    _text = ''.join(_text_list)

                    if content_idx_ - 1 >= 0 and len(dialogue_['dialog_info'][content_idx_ - 1]) < 40:
                        forward_text = dialogue_['dialog_info'][content_idx_ - 1]['sender'] + ':' + \
                                       dialogue_['dialog_info'][content_idx_ - 1]['text'] + ';'
                    else:
                        forward_text = ''

                    if contents_['sender'] == '医生':

                        if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_][
                                'sender'] + ':' + _text
                        else:
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_][
                                'sender'] + ':' + _text + ';'
                            temp_index = copy.deepcopy(content_idx_) + 1

                            speaker_flag = False
                            sen_counter = 0
                            while True:

                                if dialogue_['dialog_info'][temp_index]['sender'] == '患者':
                                    sen_counter += 1
                                    speaker_flag = True
                                    entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + \
                                                         dialogue_['dialog_info'][temp_index]['text'] + ';'

                                if sen_counter > 3:
                                    break

                                temp_index += 1
                                if temp_index >= len(dialogue_['dialog_info']):
                                    break

                    elif contents_['sender'] == '患者':
                        if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_][
                                'sender'] + ':' + _text
                        else:
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_][
                                'sender'] + ':' + _text + ';'
                            temp_index = copy.deepcopy(content_idx_) + 1

                            speaker_flag = False
                            sen_counter = 0
                            while True:

                                sen_counter += 1
                                speaker_flag = True
                                entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + \
                                                     dialogue_['dialog_info'][temp_index]['text'] + ';'

                                if sen_counter > 3:
                                    break

                                temp_index += 1
                                if temp_index >= len(dialogue_['dialog_info']):
                                    break
                    else:
                        entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_][
                            'sender'] + ':' + _text

                    if term_['name'] == 'undefined':
                        add_text = '|没有标准化'
                    else:
                        add_text = '|标准化为' + term_['name']

                    entity_['text_b'] = term_['mention'] + add_text
                    entity_['start_idx'] = term_['range'][0]
                    entity_['end_idx'] = term_['range'][1] - 1

                    entity_['label'] = term_['attr']
                    idx_ += 1

                    dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx][
                        'attr'] = tm_predictor_instance.predict_one_sample([entity_['text_a'], entity_['text_b']])
        submit_result.append(dialogue_)

    with open('../CHIP-MDCFNPC_test.jsonl', 'w', encoding="utf-8") as output_data:
        for json_content in submit_result:
            output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')
