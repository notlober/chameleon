import argparse

import torch
from PIL import Image

import cerebras.pytorch as cstorch
from model import ChameleonConfig, ChameleonModel
from prepare_mini_coco import (
    UnifiedTokenizer,
    ImageTokenizer,
    DEFAULT_IMAGE_SIZE,
    SPECIAL_TOKENS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=200)
parser.add_argument("--max_length", type=int, default=500)
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
args = parser.parse_args()

# Load the checkpoint, model config, and instantiate the model
state_dict = cstorch.load(args.checkpoint_path)
model_config = ChameleonConfig(**state_dict["model_config"])
model = ChameleonModel(model_config)
model.load_state_dict(state_dict["model"])

# Set the model to eval mode and move to GPU if available
model.eval()
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        model.bfloat16()
    model.cuda()

# Instantiate the tokenizers
tokenizer = UnifiedTokenizer()
img_tokenizer = ImageTokenizer(target_size=DEFAULT_IMAGE_SIZE)

# Function to extract image tokens from a generated sequence
def extract_image_tokens(generated_sequence):
    image_tokens = []
    collecting = False
    for token in generated_sequence:
        if token == SPECIAL_TOKENS["<|beginning-of-image|>"]:
            collecting = True
        elif token == SPECIAL_TOKENS["<|end-of-image|>"]:
            break
        elif collecting:
            image_tokens.append(token)
    return image_tokens

with torch.no_grad():
    while prompt := input("Enter a prompt (RETURN to exit): "):
        # Encode the prompt
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

        # Generate a response
        response = model(
            input_ids,
        ).cpu().squeeze().tolist()

        # Decode the response
        response_text = tokenizer.decode(response)

        # Extract image tokens if present
        image_tokens = extract_image_tokens(response)
        if image_tokens:
            image = img_tokenizer.decode(image_tokens)
            image.save("generated_image.png")
            print(f"Generated image saved to generated_image.png")

        # Print the response text
        print(f"Response: {response_text}")