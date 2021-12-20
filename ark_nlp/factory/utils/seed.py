import random
import torch
import random
import numpy as np


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