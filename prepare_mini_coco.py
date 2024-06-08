import os
import base64
import requests
import numpy as np
from PIL import Image
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

DEFAULT_VOCAB_SIZE = 256
DEFAULT_IMAGE_SIZE = (16, 16)  # an image is worth 16x16 words :)

# Special Tokens for Text and Image Segmentation
SPECIAL_TOKENS = {
    "<|beginning-of-text|>": 100264,
    "<|end-of-text|>": 100265,
    "<|beginning-of-image|>": 100266,
    "<|end-of-image|>": 100267,
    "<|user|>": 100268,
    "<|assistant|>": 100269,
    "<|eos|>": 100270,
}


class ImageTokenizer:
    """Tokenizer for encoding and decoding images."""

    def __init__(self, target_size=DEFAULT_IMAGE_SIZE):
        self.target_size = target_size

    def encode(self, image_input):
        """Encode an image into a list of RGB tokens."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        quantized_image = image.resize(self.target_size)
        quantized_array = np.array(quantized_image)

        tokens = []
        for row in range(self.target_size[0]):
            for col in range(self.target_size[1]):
                r, g, b = quantized_array[row, col]
                tokens.append(f"<R_{r}>")
                tokens.append(f"<G_{g}>")
                tokens.append(f"<B_{b}>")
        return tokens

    def decode(self, tokens, original_size=(224, 224)):
        """Decode a list of RGB tokens into an image."""
        quantized_array = np.zeros(
            (self.target_size[0], self.target_size[1], 3), dtype=np.uint8
        )
        for i, token in enumerate(tokens):
            channel, value = token[1:-1].split("_")
            row = (i // 3) // self.target_size[1]
            col = (i // 3) % self.target_size[1]
            if channel == "R":
                quantized_array[row, col, 0] = int(value)
            elif channel == "G":
                quantized_array[row, col, 1] = int(value)
            elif channel == "B":
                quantized_array[row, col, 2] = int(value)

        image = Image.fromarray(quantized_array)
        resized_image = image.resize(original_size)
        return resized_image


class UnifiedTokenizer:
    """Tokenizer for encoding and decoding both text and images."""

    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE):
        self.cl100k_base = tiktoken.get_encoding("cl100k_base")
        self.rgb_tokens = self.create_rgb_vocabulary(vocab_size)
        self.special_tokens = SPECIAL_TOKENS.copy()
        self.special_tokens.update(self.rgb_tokens)

        self.enc = tiktoken.Encoding(
            name="chameleon",
            pat_str=self.cl100k_base._pat_str,
            mergeable_ranks=self.cl100k_base._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def create_rgb_vocabulary(self, vocab_size=DEFAULT_VOCAB_SIZE):
        """Create a vocabulary of RGB tokens."""
        tokens = {}
        for channel in ["R", "G", "B"]:
            for value in range(vocab_size):
                tokens[f"<{channel}_{value}>"] = (
                    self.cl100k_base.n_vocab + len(tokens)
                )
        return tokens

    def encode(self, text):
        """Encode text or image tokens into token IDs."""
        if (
            text.startswith("<R_")
            or text.startswith("<G_")
            or text.startswith("<B_")
        ):
            return self.enc.encode(text, allowed_special="all")
        else:
            return self.enc.encode(text, allowed_special="all")

    def decode(self, token_ids):
        """Decode token IDs into text or image tokens."""
        return self.enc.decode(token_ids)


def save_vocab(url, file_name):
    """Download and save a vocabulary file from a URL."""
    res = requests.get(url)
    contents = res.content

    with open(file_name, "w", encoding="utf-8") as file:
        for line in contents.splitlines():
            if line:
                token, rank = line.split()
                decoded_token = base64.b64decode(token)

                try:
                    decoded_token_str = decoded_token.decode("utf-8")
                except UnicodeDecodeError:
                    decoded_token_str = str(decoded_token)

                file.write(f"{decoded_token_str}\n")


def create_rgb_vocabulary(
    vocab_size=DEFAULT_VOCAB_SIZE, save_path="img_tokens.txt"
):
    """Create a vocabulary of RGB tokens and save to a file."""
    tokens = []
    for channel in ["R", "G", "B"]:
        for value in range(vocab_size):
            tokens.append(f"<{channel}_{value}>")

    with open(save_path, "w") as f:
        for token in tokens:
            f.write(token + "\n")


def concat_files(file1, file2, output_file):
    """Concatenate two files into a single output file."""
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_name in [file1, file2]:
            with open(file_name, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())


def process_data(dataset, tokenizer, img_enc, process_func):
    """Tokenize and process a dataset."""
    return dataset.map(
        process_func,
        fn_kwargs={"tokenizer": tokenizer, "img_enc": img_enc},
        desc="tokenizing the splits",
    )


def save_data_to_bin(tokenized_dataset, filename, dtype=np.uint32, total_batches=32): # uint32 because uint16 cant pass 100k n_vocab
    """Save tokenized data to a binary file."""
    arr_len = np.sum(tokenized_dataset["len"], dtype=np.uint64)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = tokenized_dataset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])

        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()


def process_image_then_text(example, tokenizer, img_enc):
    """Process data for image-to-text generation."""
    ids = []

    ids.extend(tokenizer.encode("<|user|>")) 
    ids.extend(tokenizer.encode("<|beginning-of-image|>"))
    image_tokens = img_enc.encode(example["image"])
    for image_token in image_tokens:
        ids.extend(tokenizer.encode(image_token))
    ids.extend(tokenizer.encode("<|end-of-image|>"))
    ids.extend(tokenizer.encode("<|assistant|>")) 
    ids.extend(tokenizer.encode("<|beginning-of-text|>"))
    ids.extend(tokenizer.encode(example["caption"][0]))
    ids.extend(tokenizer.encode("<|end-of-text|>")) 
    ids.extend(tokenizer.encode("<|eos|>")) 

    out = {"ids": ids, "len": len(ids)}
    return out


def process_text_then_image(example, tokenizer, img_enc):
    """Process data for text-to-image generation."""
    ids = []

    ids.extend(tokenizer.encode("<|user|>")) 
    ids.extend(tokenizer.encode("<|beginning-of-text|>"))
    ids.extend(tokenizer.encode(example["caption"][0]))
    ids.extend(tokenizer.encode("<|end-of-text|>"))
    ids.extend(tokenizer.encode("<|assistant|>")) 
    ids.extend(tokenizer.encode("<|beginning-of-image|>"))
    image_tokens = img_enc.encode(example["image"])
    for image_token in image_tokens:
        ids.extend(tokenizer.encode(image_token))
    ids.extend(tokenizer.encode("<|end-of-image|>")) 
    ids.extend(tokenizer.encode("<|eos|>")) 

    out = {"ids": ids, "len": len(ids)}
    return out

if __name__ == "__main__":
    # Download and save vocabulary files
    save_vocab(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "gpt4_gpt3.5turbo_vocab.txt",
    )
    create_rgb_vocabulary()

    # Merge text and image vocabularies
    concat_files(
        "gpt4_gpt3.5turbo_vocab.txt",
        "img_tokens.txt",
        "combined_vocab_gpt4_img.txt",
    )

    # Load dataset
    dataset = load_dataset(
        "nlphuji/mscoco_2014_5k_test_image_text_retrieval", split="test[:50]"
    )

    # Instantiate tokenizers
    tokenizer = UnifiedTokenizer()
    img_enc = ImageTokenizer(target_size=DEFAULT_IMAGE_SIZE)

    # Choose data processing function based on desired input/output
    # process_func = process_image_then_text  # For image-to-text
    process_func = process_text_then_image  # For text-to-image

    # Process and save data
    tokenized_dataset = process_data(dataset, tokenizer, img_enc, process_func)
    save_data_to_bin(tokenized_dataset, "train.bin")

    print("Data preparation completed successfully!")