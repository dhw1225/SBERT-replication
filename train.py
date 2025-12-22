import jittor as jt
from models.siamese_network import SiameseSBERT
from dataloader import NLIDataset
from transformers import AutoTokenizer
from data_utils import load_nli_data

def train():
    jt.flags.use_cuda = 0
    
    model_path = "models/roberta-base"
    model = SiameseSBERT(model_path)
    optimizer = jt.optim.Adam(model.parameters(), lr=2e-5) # 设定 lr=2e-5
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nli_data_paths = [
        'data/snli_1.0/snli_1.0_train.jsonl',         # SNLI 训练集
        'data/multinli_1.0/multinli_1.0_train.jsonl'  # MultiNLI 训练集
    ]
    train_samples = load_nli_data(nli_data_paths, limit=1000)
    train_dataset = NLIDataset(train_samples, tokenizer)
    train_loader = train_dataset.set_attrs(batch_size=16, shuffle=True, collate_batch=train_dataset.collate_batch)

    # 训练循环（仅 1 Epoch）
    model.train()
    for batch_idx, (in_a, in_b, labels) in enumerate(train_loader):
        loss, logits = model(in_a, in_b, labels)
        
        optimizer.step(loss) # 更新梯度
        
        if batch_idx % 100 == 0:
            print(f"Epoch 1, Step {batch_idx}, Loss: {loss.item():.4f}")

    # 保存模型
    model.save("data/sbert_jittor_roberta.pkl")

if __name__ == "__main__":
    train()