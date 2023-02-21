from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup # AdamW
from tqdm import tqdm, trange

# train csv file to result

# 전처리
def read_preprocess_csv(filepath):
    df = pd.read_csv(filepath)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    breakpoint()

filepath = 'data_csv/2021-06(named,alpha)_v2.csv'
read_preprocess_csv(filepath)