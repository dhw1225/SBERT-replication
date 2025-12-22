import os
import json

def load_nli_data(file_paths, limit=None):
    # entailment: 0, neutral: 1, contradiction: 2
    label_map = {
        'entailment': 0, 
        'neutral': 1, 
        'contradiction': 2
    }
    
    all_samples = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not exist {file_path}!")
            continue
            
        print(f"Loading data from {file_path} ...")
        
        with open(file_path, 'rt', encoding='utf-8') as f:
            count = 0
            for line in f:
                row = json.loads(line)
                label = row.get('gold_label')
                sent1 = row.get('sentence1')
                sent2 = row.get('sentence2')
                    
                # 过滤掉标签为 '-' (无共识) 的数据
                if label in label_map and sent1 and sent2:
                    all_samples.append((sent1, sent2, label_map[label]))
                    count += 1
                
                if limit is not None and count >= limit:
                    break

    print(f"Data loading complete. Total valid samples: {len(all_samples)}")
    return all_samples