import jittor as jt
import jittor.nn as nn
from models.modeling_jittor import JittorBertModel, JittorConfig
from transformers import AutoModel, AutoConfig

def load_hf_weights(jit_model, hf_model_name):
    print(f"Loading weights from {hf_model_name} to Jittor model...")
    
    # 加载 PyTorch 模型，用来提取 state_dict
    pt_model = AutoModel.from_pretrained(hf_model_name)
    pt_state = pt_model.state_dict()
    # 获取 Jittor 模型的参数字典
    jit_state = jit_model.state_dict()

    loaded_cnt = 0
    
    for key, param in jit_state.items():
        # 尝试 1: 直接匹配
        pt_key = key
        
        # 尝试 2: 加上 bert. 前缀
        if pt_key not in pt_state:
            pt_key = "bert." + key
            
        # 尝试 3: 加上 roberta. 前缀
        if pt_key not in pt_state:
            pt_key = "roberta." + key
            
        # 尝试 4 (特殊情况): 遍历 pt_state 寻找结尾匹配的 key
        if pt_key not in pt_state:
            for potential_key in pt_state.keys():
                if potential_key.endswith(key):
                    pt_key = potential_key
                    break
        
        if pt_key in pt_state:
            pt_np = pt_state[pt_key].cpu().detach().numpy()
            
            if param.shape != pt_np.shape:
                # 处理 Linear 层转置问题 (PyTorch Linear 是 [out, in]，Jittor 是 [in, out])
                if len(param.shape) == 2 and len(pt_np.shape) == 2:
                    if param.shape == pt_np.T.shape:
                        pt_np = pt_np.T
                    else:
                        print(f"Skipping {key}: Shape mismatch {param.shape} vs {pt_np.shape}")
                        continue
                else:
                    print(f"Skipping {key}: Shape mismatch {param.shape} vs {pt_np.shape}")
                    continue
            
            param.assign(pt_np)
            loaded_cnt += 1
        else:
            pass

    print(f"Successfully loaded {loaded_cnt} parameters layers.")

class SBERTModel(nn.Module):
    def __init__(self, model_path='models/bert-large-uncased', pooling='mean'):
        super(SBERTModel, self).__init__()
        
        # 根据 model_name 读取配置
        hf_config = AutoConfig.from_pretrained(model_path)
        
        # 将 HF 配置转为 Jittor 配置
        jt_config = JittorConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=hf_config.type_vocab_size,
            layer_norm_eps=hf_config.layer_norm_eps
        )
        
        # 初始化 Jittor 版 BERT
        self.transformer = JittorBertModel(jt_config)
        
        # 加载权重
        load_hf_weights(self.transformer, model_path)
        
        self.pooling = pooling
        self.hidden_size = hf_config.hidden_size
        
    def execute(self, input_ids, attention_mask=None, token_type_ids=None):
        # transformer 返回 (last_hidden_state, all_hidden_states)
        outputs, _ = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        if self.pooling == 'cls':
            # CLS Pooling
            embeddings = outputs[:, 0, :]
        
        elif self.pooling == 'mean':
            # Mean Pooling
            # [Batch, Seq] -> [Batch, Seq, Hidden]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs).float32()
            
            # 对 embeddings 进行加权求和（只保留非 Padding 部分）
            sum_embeddings = jt.sum(outputs * input_mask_expanded, dim=1)
            
            # 计算非 Padding 的 token 数量（加一个极小值 1e-9 防止除以 0）
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = jt.clamp(sum_mask, min_v=1e-9)
            embeddings = sum_embeddings / sum_mask
                
        else:
            # Max Pooling
            if attention_mask is not None:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs).float32()
                outputs[input_mask_expanded == 0] = -1e9 # Mask 掉的部分设为极小值
            
            embeddings = jt.max(outputs, dim=1)
            
        return embeddings