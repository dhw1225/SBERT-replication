import jittor as jt
from transformers import AutoTokenizer
from models.siamese_network import SiameseSBERT

class SentenceEncoder:
    """Sentence-level encoder that exposes an encode(sentences) interface."""

    def __init__(self, siamese_model, model_path, batch_size=32, max_length=128):
        self.model = siamese_model
        self.sbert = siamese_model.sbert  # SBERTModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.eval()

    def eval(self):
        # Keep compatibility with code that expects a nn.Module-like interface
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )

            input_ids = jt.array(enc["input_ids"]).int32()
            attention_mask = jt.array(enc["attention_mask"]).int32()

            token_type_ids = enc.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = jt.array(token_type_ids).int32()

            with jt.no_grad():
                embeddings = self.sbert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            all_embeddings.append(embeddings)

        return jt.concat(all_embeddings, dim=0) if len(all_embeddings) > 1 else all_embeddings[0]


def load_sbert_model(
    checkpoint_path,
    model_path,
    pooling="mean",
    num_labels=3,
    strict=True
):
    """
    Load a trained SiameseSBERT model from .pkl and return a sentence encoder.
    """

    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = jt.load(checkpoint_path)

    model = SiameseSBERT(
        model_path=model_path,
        pooling=pooling,
        num_labels=num_labels
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully")

    return SentenceEncoder(model, model_path=model_path)