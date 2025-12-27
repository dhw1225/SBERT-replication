from datasets import load_dataset

def load_sts_dataset(hf_name):
    """
    Load an STS-style dataset from HuggingFace and return:
    - sentence_pairs: List[(sentence1, sentence2)]
    - scores: List[float]
    """

    # # STSb has validation + test; paper uses test
    # if hf_name == "sentence-transformers/stsb":
    #     split = "test"
    # else:
    #     split = "test"
    
    split="test"

    ds = load_dataset(hf_name, split=split)

    sentence_pairs = []
    scores = []

    for item in ds:
        sentence_pairs.append((item["sentence1"], item["sentence2"]))
        scores.append(float(item["score"]))

    return sentence_pairs, scores