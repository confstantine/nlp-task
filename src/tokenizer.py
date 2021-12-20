import unicodedata
from abc import ABCMeta
from copy import deepcopy
import numpy as np
from transformers import AutoTokenizer


class BaseTokenizer(object, metaclass=ABCMeta):
    def __init__(self, vocab, max_seq_len):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def tokenize(self, text):
        return self.vocab.tokenize(text)

    def pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x


class TransfomerTokenizer(BaseTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """

    def __init__(self, vocab, max_seq_len):

        if isinstance(vocab, str):
            # TODO: 改成由自定义的字典所决定
            vocab = AutoTokenizer.from_pretrained(vocab)

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.additional_special_tokens = set()
        self.tokenizer_type = 'transfomer'

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def recover_bert_token(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def get_token_mapping(self, text, tokens, is_mapping_index=True):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        raw_text = deepcopy(text)
        text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            token = token.lower()
            if token == '[unk]' or token in self.additional_special_tokens:
                if is_mapping_index:
                    token_mapping.append(char_mapping[offset:offset + 1])
                else:
                    token_mapping.append(raw_text[offset:offset + 1])
                offset = offset + 1
            elif self._is_special(token):
                token_mapping.append([])  # 如果是[CLS]或者是[SEP]之类的词，则没有对应的映射
            else:
                token = self.recover_bert_token(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                if is_mapping_index:
                    token_mapping.append(char_mapping[start:end])
                else:
                    token_mapping.append(raw_text[start:end])
                offset = end

        return token_mapping

    def sequence_to_ids(self, sequence_a, sequence_b=None):
        if sequence_b is None:
            return self.sentence_to_ids(sequence_a)
        else:
            return self.pair_to_ids(sequence_a, sequence_b)

    def sentence_to_ids(self, sequence, return_sequence_length=False):
        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        if return_sequence_length:
            sequence_length = len(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 2:
            sequence = sequence[0:(self.max_seq_len - 2)]

        speaker_ids = []
        id_ = 0
        for idx_, term_ in enumerate(sequence):
            if term_ == '医' and sequence[idx_ + 1] == '生':
                id_ = 1
            if term_ == '患' and sequence[idx_ + 1] == '者':
                id_ = 2

            speaker_ids.append(id_)

        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence + ['[SEP]']
        speaker_ids = [0] + speaker_ids + [0]
        segment_ids = [0] * len(sequence)

        e11_p = sequence.index("<") + 1
        e12_p = sequence.index(">") - 1

        e1_mask = [0] * len(sequence)
        for _i in range(e11_p, e12_p + 1):
            e1_mask[_i] = 1
            break

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding
        e1_mask += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')
        e1_mask = np.asarray(e1_mask, dtype='int64')

        if return_sequence_length:
            return sequence, sequence_mask, segment_ids, e1_mask, sequence_length

        return sequence, sequence_mask, segment_ids, e1_mask

    def pair_to_ids(self, sequence_a, sequence_b, return_sequence_length=False):

        raw_sequence_a = deepcopy(sequence_a)

        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a)

        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b)

        if return_sequence_length:
            sequence_length = (len(sequence_a), len(sequence_b))

        # 对超长序列进行截断
        start_idx = 0
        end_idx = self.max_seq_len - len(sequence_b) - 3
        entity_end_idx = sequence_a.index('[unused2]')
        end_idx = entity_end_idx + 20
        if end_idx < (self.max_seq_len - len(sequence_b)):
            sequence_a = sequence_a[0:(self.max_seq_len - len(sequence_b)) - 3]
        else:
            end_idx = end_idx - 20 + (self.max_seq_len - len(sequence_b)) / 2
            start_idx = end_idx - (self.max_seq_len - len(sequence_b)) + 3
            if start_idx < 0:
                start_idx = 0
            sequence_a = sequence_a[int(start_idx):int(end_idx)]

        #         sequence_a = sequence_a[0:(self.max_seq_len - len(sequence_b))]
        #         if len(sequence_a) > ((self.max_seq_len - 3)//2):
        #             sequence_a = sequence_a[0:(self.max_seq_len - 3)//2]
        #         if len(sequence_b) > ((self.max_seq_len - 3)//2):
        #             sequence_b = sequence_b[0:(self.max_seq_len - 3)//2]

        speaker_ids = [0]
        id_ = 0
        for idx_, term_ in enumerate(sequence_a):
            try:
                if term_ == '医' and idx_ < len(sequence_a) - 1 and sequence_a[idx_ + 1] == '生':
                    id_ = 1
                if term_ == '患' and idx_ < len(sequence_a) - 1 and sequence_a[idx_ + 1] == '者':
                    id_ = 2
            except:
                print(sequence_a)
                print(idx_)

            speaker_ids.append(id_)

        speaker_ids.append(0)
        for idx_, term_ in enumerate(sequence_b):
            speaker_ids.append(3)
        speaker_ids.append(0)

        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence_a + ['[SEP]'] + sequence_b + ['[SEP]']
        segment_ids = [0] * (len(sequence_a) + 2) + [1] * (len(sequence_b) + 1)
        try:
            e11_p = sequence.index("[unused1]") + 1
            e12_p = sequence.index("[unused2]") - 1
        except:
            print(raw_sequence_a)
            print(sequence_a)

        e1_mask = [0] * len(sequence)
        for _i in range(e11_p, e12_p + 1):
            e1_mask[_i] = 1

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        speaker_ids += padding
        e1_mask += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')
        speaker_ids = np.asarray(speaker_ids, dtype='int64')
        e1_mask = np.asarray(e1_mask, dtype='int64')

        #         if len(sequence) > 150:
        #             print('sequence', raw_sequence_a)
        #         if len(sequence_mask) > 150:
        #             print(len(sequence_mask))
        #             print(len(sequence))
        #             print('sequence_mask', raw_sequence_a)
        #         if len(segment_ids) > 150:
        #             print('segment_ids', raw_sequence_a)
        #         if len(speaker_ids) > 150:
        #             print('speaker_ids', raw_sequence_a)
        #         if len(e1_mask) > 150:
        #             print('e1_mask', raw_sequence_a)

        if return_sequence_length:
            return sequence, sequence_mask, segment_ids, speaker_ids, e1_mask, sequence_length

        return sequence, sequence_mask, segment_ids, speaker_ids, e1_mask