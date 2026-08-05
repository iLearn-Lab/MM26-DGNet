"""Export the two fixed CLIP text features used by DGNet's PWM modules."""

import argparse
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer


TARGET_PROMPT = "an infrared image containing small and sparse thermal targets"
BACKGROUND_PROMPT = "an infrared image occupied by large and smooth background regions"


def encode_prompt(text_encoder, tokenizer, prompt):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=30,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )
    return output.pooler_output.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="./clip-vit-base-patch32")
    parser.add_argument("--output", default="./weights/dgnet_text_features.pth")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"CLIP text model directory not found: {model_path}")

    tokenizer = CLIPTokenizer.from_pretrained(str(model_path))
    text_encoder = CLIPTextModel.from_pretrained(str(model_path)).eval()
    text_encoder.requires_grad_(False)
    features = {
        "target": encode_prompt(text_encoder, tokenizer, TARGET_PROMPT),
        "background": encode_prompt(text_encoder, tokenizer, BACKGROUND_PROMPT),
        "target_prompt": TARGET_PROMPT,
        "background_prompt": BACKGROUND_PROMPT,
        "encoder": "CLIP-ViT-B/32",
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    print(f"Saved fixed DGNet text features to {output_path}")


if __name__ == "__main__":
    main()
