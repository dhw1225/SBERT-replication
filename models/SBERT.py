import jittor as jt
import jittor.nn as nn
from transformers import BertModel, RobertaModel

class SBERTModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling='mean'):
        super(SBERTModel, self).__init__()
        # 加载预训练模型
        if 'roberta' in model_name.lower():
            self.bert = RobertaModel.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        
        self.pooling = pooling
        self.hidden_size = self.bert.config.hidden_size
        
    def execute(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 池化策略
        if self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            if attention_mask is not None:
                # 计算有效token的平均值
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = jt.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                sum_mask = jt.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                embeddings = outputs.last_hidden_state.mean(1)
        elif self.pooling == 'max':
            # 最大池化
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            outputs.last_hidden_state[input_mask_expanded == 0] = -1e9
            embeddings = outputs.last_hidden_state.max(1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
            
        return embeddings