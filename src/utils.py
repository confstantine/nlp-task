import random
import torch
import codecs
import json
import copy
import numpy as np
import pandas as pd
from torch.optim import *
from torch.optim import Optimizer
from transformers import AdamW
from label_smoothing import LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, SmoothL1Loss

all_losses_dict = dict(bce=BCEWithLogitsLoss,
                       ce=CrossEntropyLoss,
                       smoothl1=SmoothL1Loss,
                       lsce=LabelSmoothingCrossEntropy,
                       )

all_optimizers_dict = dict(
    adadelta=Adadelta,
    adagrad=Adagrad,
    adam=Adam,
    sparseadam=SparseAdam,
    adamax=Adamax,
    asgd=ASGD,
    lbfgs=LBFGS,
    rmsprop=RMSprop,
    rprop=Rprop,
    sgd=SGD,
    adamw=AdamW)


def set_seed(seed):
    """
    设置随机种子
    :param seed:

    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_task_data(data_path):
    with codecs.open(data_path, mode='r', encoding='utf8') as f:
        reader = f.readlines(f)

    data_list = []

    for dialogue_ in reader:
        dialogue_ = json.loads(dialogue_)

        _dialog_id = dialogue_['dialog_id']

        for content_idx_, contents_ in enumerate(dialogue_['dialog_info']):

            terms_ = contents_['ner']

            if len(terms_) != 0:
                idx_ = 0
                for _, term_ in enumerate(terms_):

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
                    entity_['text_b_copy'] = term_['mention']
                    entity_['start_idx'] = term_['range'][0]
                    entity_['end_idx'] = term_['range'][1] - 1

                    try:
                        entity_['label_b'] = term_['name']
                    except:
                        print(contents_)
                        print(term_)
                    entity_['label'] = term_['attr']
                    entity_['dialog_id'] = _dialog_id
                    idx_ += 1

                    if entity_['label'] == '':
                        continue

                    if len(entity_) == 0:
                        continue

                    data_list.append(entity_)

    data_df = pd.DataFrame(data_list)

    data_df = data_df.loc[:,
              ['dialog_id', 'text_b_copy', 'text_a', 'text_b', 'start_idx', 'end_idx', 'label_b', 'label', 'dialogue']]

    return data_df


def random_split_train_and_dev(data_df, split_rate=0.9):
    data_df = data_df.sample(frac=1, random_state=42)
    train_size = int(split_rate * len(data_df))
    train_df = data_df[:train_size]
    dev_df = data_df[train_size:]

    return train_df, dev_df


def get_optimizer(optimizer, module, lr=False, params=None):
    if params is None:
        params_ = (p for p in module.parameters() if p.requires_grad)
    else:
        params_ = params

    if isinstance(optimizer, str):
        optimizer = all_optimizers_dict[optimizer](params_)
    elif type(optimizer).__name__ == 'type' and issubclass(optimizer, Optimizer):
        optimizer = optimizer(params_)
    elif isinstance(optimizer, Optimizer):
        if params is not None:
            optimizer.param_groups = params
    else:
        raise ValueError("The optimizer type does not exist")

    if lr is not False:
        for param_groups_ in optimizer.param_groups:
            param_groups_['lr'] = lr

    return optimizer


def get_loss(loss_):
    if isinstance(loss_, str):
        loss_ = loss_.lower()
        loss_ = loss_.replace('_', '')
        return all_losses_dict[loss_]()

    return loss_
