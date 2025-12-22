import jittor as jt
import jittor.nn as nn
from SBERT import SBERTModel

class SiameseSBERT(nn.Module):
    def __init__(self, model_name='bert-large-uncased', pooling='mean', num_labels=3):
        super(SiameseSBERT, self).__init__()
        self.model = SBERTModel(model_name, pooling)
        self.classifier = nn.Linear(self.model.hidden_size * 3, num_labels)
        
    def get_sentence_embedding(self, input_ids, attention_mask):
        # transformer 输出是一个元组，第一个元素是 last_hidden_state
        outputs = self.transformer(input_ids, attention_mask)
        token_embeddings = outputs[0] 
        embedding = self.pooling(token_embeddings, attention_mask)
        return embedding

    def execute(self, inputs_a, inputs_b, labels=None):
        # 生成句向量 u 和 v
        u = self.get_sentence_embedding(inputs_a['input_ids'], inputs_a['attention_mask'])
        v = self.get_sentence_embedding(inputs_b['input_ids'], inputs_b['attention_mask'])
        # 拼接 (u, v, |u-v|)
        abs_diff = jt.abs(u - v)
        features = jt.contrib.concat([u, v, abs_diff], dim=1)
        # 输出 logits，计算 loss
        logits = self.classifier(features)
        if labels is not None:
            loss = nn.cross_entropy_loss(logits, labels)
            return loss, logits
        return logits