"""
Step 4: Generate question embeddings for the coding teacher model.

Uses a frozen LLM backbone (e.g., Qwen2.5-Coder-1.5B-Instruct) to encode
all questions and save as a .pt file, aligned with questions_coding_10240.parquet.

Usage:
    python save_coding_embeddings.py \
        --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --questions_json ../datasets/coding/coding_questions.json \
        --output_dir ../datasets/coding/embeddings
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def encode_questions(
    model_name: str,
    questions: list[str],
    batch_size: int = 32,
    max_length: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode all questions using the model backbone with mean pooling."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    padding_side = "left" if "qwen" in model_name.lower() else "right"
    tokenizer.padding_side = padding_side

    all_embeddings = []
    dataset = TextDataset(questions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_texts in tqdm(dataloader, desc="Encoding questions"):
        encoded = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[-1]

            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            all_embeddings.append(pooled.cpu().float())

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate question embeddings")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--questions_json", type=str, required=True,
                        help="Path to coding_questions.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import json
    with open(args.questions_json) as f:
        questions_data = json.load(f)
    questions = [q["question"] for q in questions_data]
    print(f"Loaded {len(questions)} questions")

    embeddings = encode_questions(
        args.model_name, questions,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    safe_name = args.model_name.replace("/", "_")
    save_path = os.path.join(args.output_dir, f"{safe_name}.pt")
    torch.save(embeddings, save_path)
    print(f"Saved embeddings to {save_path}")

    print("[DONE] save_coding_embeddings complete.")


if __name__ == "__main__":
    main()
