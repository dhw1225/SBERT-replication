import jittor as jt
from jittor.dataset import Dataset
import numpy as np

class NLIDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        super().__init__()
        self.samples = samples # 格式: [(text_a, text_b, label), ...]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.set_attrs(total_len=len(samples))

    def __getitem__(self, idx):
        text_a, text_b, label = self.samples[idx]
        return text_a, text_b, label

    def collate_batch(self, batch):
        # 处理 batch 内的 tokenization 和 padding
        texts_a = [item[0] for item in batch]
        texts_b = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        encoding_a = self.tokenizer(texts_a, padding=True, truncation=True, 
                                    max_length=self.max_len, return_tensors='np')
        encoding_b = self.tokenizer(texts_b, padding=True, truncation=True, 
                                    max_length=self.max_len, return_tensors='np')
        
        in_a = {
            'input_ids': jt.array(encoding_a['input_ids']),
            'attention_mask': jt.array(encoding_a['attention_mask'])
        }
        in_b = {
            'input_ids': jt.array(encoding_b['input_ids']),
            'attention_mask': jt.array(encoding_b['attention_mask'])
        }
        labels = jt.array(np.array(labels))
        
        return in_a, in_b, labels