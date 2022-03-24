

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# NSMC 데이터 처리
from Korpora import Korpora
Korpora.fetch("nsmc")

corpus = Korpora.load("nsmc")

df_train_text = pd.DataFrame(corpus.train.texts, columns=['text'])
df_train_labels = pd.DataFrame(corpus.train.labels, columns=['labels'])
df_train = pd.concat([df_train_text, df_train_labels], axis=1)

df_test_text = pd.DataFrame(corpus.test.texts, columns=['text'])
df_test_labels = pd.DataFrame(corpus.test.labels, columns=['labels'])
df_test = pd.concat([df_test_text, df_test_labels], axis=1)

class NSMCDataset(Dataset):
    def __init__(self, tokenizer, df, text_labels, max_length=512):
        self.tokenizer = tokenizer
        self.df = df
        self.text_labels = text_labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)# // 10
    
    def __getitem__(self, index):
        input_ids = self.tokenizer(self.df.iloc[index]['text'], padding=True, truncation=True, max_length=self.max_length, add_special_tokens=True)
        labels = self.tokenizer(self.text_labels[self.df.iloc[index]['labels']])

        return { 'input_ids': torch.tensor(input_ids['input_ids']), 'labels': torch.tensor(labels['input_ids']) }

def get_dataset(tokenizer, name='train'):
    if name == 'train':
        df = df_train
    else:
        df = df_test
    return NSMCDataset(tokenizer, df, ['negative', 'positive'], max_length=512)

def nsmc_metric(tokenizer, pred):
    n_correct, n_incorrect = 0, 0
    for i in range(len(pred.predictions)):
        output = tokenizer.decode(pred.predictions[i], skip_special_tokens=True)
        target = tokenizer.decode(pred.label_ids[i], skip_special_tokens=True)
        if output == target:
            n_correct += 1
        else:
            n_incorrect += 1

    return { "nsmc_accuracy": n_correct / (n_correct + n_incorrect) }

# import json
# import numpy as np
# from typing import List, Dict, Callable, Tuple
# from sklearn.metrics import f1_score
# from transformers import EvalPrediction, PreTrainedTokenizer
# from seq2seq.utils import (lmap,)

# # label2idx = {"정치": 0, "경제": 1, "사회": 2, "생활문화": 3, "세계": 4, "IT과학": 5, "스포츠": 6}
# ynat_label_list = ["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"]

# def build_compute_metrics_fn_et5(tokenizer: PreTrainedTokenizer, _label_list=ynat_label_list) -> Callable[[EvalPrediction], Dict]:
#     label_list = _label_list

#     def non_pad_len(tokens: np.ndarray) -> int:
#         return np.count_nonzero(tokens != tokenizer.pad_token_id)

#     def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
#         pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
#         label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
#         pred_str = lmap(str.strip, pred_str)
#         label_str = lmap(str.strip, label_str)
#         return pred_str, label_str

#     def ynat_metrics(pred: EvalPrediction) -> Dict:
#         pred_str, label_str = decode_pred(pred)
#         pred_str = [x if x in label_list else 'none' for x in pred_str]
#         label_str = [x if x in label_list else 'none' for x in label_str]
#         result = f1_score(y_true=label_str, y_pred=pred_str, average='macro')

#         return {
#             "F1(macro)": result,
#         }

#     compute_metrics_fn = ynat_metrics
#     return compute_metrics_fn

if __name__ == "__main__":
    ds = get_dataset(name='train')
    loader = DataLoader(ds, batch_size=1)
    for i, batch in enumerate(loader):
        print(batch)
        if i == 5:
            break
    # for i in range(10):
    #     print(next(iter(ds)))
    print('Done.')
