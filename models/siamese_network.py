import jittor as jt
import jittor.nn as nn
from SBERT import SBERTModel

class SiameseSBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling='mean'):
        super(SiameseSBERT, self).__init__()
        self.model = SBERTModel(model_name, pooling)
        
    def execute(self, batch):
        # batch 包含一对句子的数据
        emb1 = self.model(batch['input_ids1'], batch['attention_mask1'], 
                          batch.get('token_type_ids1', None))
        emb2 = self.model(batch['input_ids2'], batch['attention_mask2'], 
                          batch.get('token_type_ids2', None))
        return emb1, emb2


class ClassificationSBERT(SiameseSBERT):
    def __init__(self, model_name='bert-base-uncased', pooling='mean', num_labels=3):
        super().__init__(model_name, pooling)
        self.classifier = nn.Linear(self.model.hidden_size * 3, num_labels)
        
    def execute(self, batch):
        u, v = super().execute(batch)
        diff = jt.abs(u - v)
        features = jt.concat([u, v, diff], dim=1)
        logits = self.classifier(features)
        return logits