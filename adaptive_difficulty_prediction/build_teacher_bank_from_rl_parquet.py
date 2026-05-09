"""Build a teacher embedding bank directly from an RL parquet dataset.

This keeps existing teacher assets untouched by writing a separate bank directory:

    <output_root>/datasets/questions_<bank_name>.parquet
    <output_root>/embeddings/<model_stem>.pt

The generated questions parquet is deduplicated by exact question string while
preserving first-seen order so teacher-side string lookup stays deterministic.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from load_data import get_embedding_stem


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def extract_questions(rl_parquet_path: str) -> list[str]:
    df = pd.read_parquet(rl_parquet_path)
    questions: list[str] = []
    seen: set[str] = set()

    for idx, row in df.iterrows():
        question = None

        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            question = extra_info.get("question")

        if question is None:
            prompt = row.get("prompt")
            if isinstance(prompt, list):
                for message in reversed(prompt):
                    if isinstance(message, dict) and message.get("role") == "user":
                        question = message.get("content")
                        if question is not None:
                            break

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Could not extract a valid question from row {idx}.")

        if question not in seen:
            seen.add(question)
            questions.append(question)

    return questions


def save_questions_parquet(questions: list[str], output_path: str, overwrite: bool = False) -> None:
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing questions parquet: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"problem": questions}).to_parquet(output_path, index=False)
    print(f"Saved questions parquet: {output_path} ({len(questions)} rows)")


def encode_questions(
    model_name: str,
    questions: list[str],
    batch_size: int,
    device: str,
    left_padding: bool,
    max_length: int | None,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if left_padding:
        tokenizer.padding_side = "left"

    model_kwargs = {"trust_remote_code": True}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()

    all_embeddings = []
    dataloader = DataLoader(TextDataset(questions), batch_size=batch_size, shuffle=False)

    for batch_texts in tqdm(dataloader, desc="Encoding questions"):
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length

        encoded = tokenizer(list(batch_texts), **tokenizer_kwargs)
        encoded = {key: value.to(device) for key, value in encoded.items()}

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
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    return embeddings


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a teacher bank from an RL parquet dataset")
    parser.add_argument("--rl_parquet", type=str, required=True, help="Path to the RL parquet dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output bank directory (default: <script_dir>/<bank_name>_teacher_bank)")
    parser.add_argument("--bank_name", type=str, default=None,
                        help="Bank name used in questions_<bank_name>.parquet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=None,
                        help="Optional tokenizer max_length; uses tokenizer default when omitted")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for encoding, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--left_padding", action="store_true",
                        help="Use left padding when encoding questions")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing generated files inside the output bank")
    args = parser.parse_args()

    rl_parquet_path = os.path.abspath(args.rl_parquet)
    bank_name = args.bank_name or os.path.splitext(os.path.basename(rl_parquet_path))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.abspath(args.output_root or os.path.join(script_dir, f"{bank_name}_teacher_bank"))
    device = resolve_device(args.device)

    print(f"RL parquet: {rl_parquet_path}")
    print(f"Bank name: {bank_name}")
    print(f"Output root: {output_root}")
    print(f"Model name: {args.model_name}")
    print(f"Encoding device: {device}")

    questions = extract_questions(rl_parquet_path)
    print(f"Unique questions extracted: {len(questions)}")

    datasets_dir = os.path.join(output_root, "datasets")
    embeddings_dir = os.path.join(output_root, "embeddings")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    questions_parquet_path = os.path.join(datasets_dir, f"questions_{bank_name}.parquet")
    embeddings_path = os.path.join(embeddings_dir, f"{get_embedding_stem(args.model_name)}.pt")

    save_questions_parquet(questions, questions_parquet_path, overwrite=args.overwrite)

    if os.path.exists(embeddings_path) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing embeddings file: {embeddings_path}")

    embeddings = encode_questions(
        model_name=args.model_name,
        questions=questions,
        batch_size=args.batch_size,
        device=device,
        left_padding=args.left_padding,
        max_length=args.max_length,
    )
    torch.save(embeddings, embeddings_path)
    print(f"Saved embeddings: {embeddings_path}")
    print("[DONE] Teacher bank generation complete.")


if __name__ == "__main__":
    main()
