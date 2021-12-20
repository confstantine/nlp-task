import copy
import pandas as pd
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame


class BaseDataset(Dataset):
    def __init__(
            self,
            data,
            categories=None,
            is_retain_dataset=False,
            is_train=True,
            is_test=False
    ):

        self.is_test = is_test
        self.is_train = is_train

        if self.is_test is True:
            self.is_train = False

        if isinstance(data, DataFrame):
            if 'label' in data.columns:
                data['label'] = data['label'].apply(lambda x: str(x))

            self.dataset = self._convert_to_dataset(data)
        else:
            self.dataset = self._load_dataset(data)

        if categories == None:
            self.categories = self._get_categories()
        else:
            self.categories = categories

        self.cat2id = dict(zip(self.categories, range(len(self.categories))))
        self.id2cat = dict(zip(range(len(self.categories)), self.categories))

        self.class_num = len(self.cat2id)

        self.is_retain_dataset = is_retain_dataset

    def _get_categories(self):
        pass

    def _read_data(self, data_path, data_format=None):
        """
        读取所需数据

        :param data_path: (string) 数据所在路径
        :param data_format: (string) 数据存储格式
        """
        if data_format == None:
            data_format = data_path.split('.')[-1]

        if data_format == 'csv':
            data_df = pd.read_csv(data_path, dtype={'label': str})
        elif data_format == 'json':
            try:
                data_df = pd.read_json(data_path, dtype={'label': str})
            except:
                data_df = self.read_line_json(data_path)
        elif data_format == 'tsv':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        elif data_format == 'txt':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        else:
            raise ValueError("The data format does not exist")

        return data_df

    def _convert_to_dataset(self, data_df):
        pass

    def _load_dataset(self, data_path):
        """
        加载数据集

        :param data_path: (string) the data file to load
        """
        data_df = self._read_data(data_path)

        return self._convert_to_dataset(data_df)

    def _get_input_length(self, text, bert_tokenizer):
        pass

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        pass

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        pass

    def _convert_to_customized_ids(self, customized_tokenizer):
        pass

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式

        :param tokenizer:
        """
        if tokenizer.tokenizer_type == 'vanilla':
            features = self._convert_to_vanilla_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'transfomer':
            features = self._convert_to_transfomer_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(tokenizer)
        else:
            raise ValueError("The tokenizer type does not exist")

        if self.is_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

        self.dataset = features

    @property
    def dataset_cols(self):
        return list(self.dataset[0].keys())

    @property
    def to_device_cols(self):
        return list(self.dataset[0].keys())

    @property
    def sample_num(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


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

            # input_a_length = self._get_input_length(row_['text_a'], bert_tokenizer)
            # input_b_length = self._get_input_length(row_['text_b'], bert_tokenizer)

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
