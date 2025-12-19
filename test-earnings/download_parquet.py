#!/usr/bin/env python3
"""Download parquet files from Hugging Face dataset."""

import urllib.request
import os

BASE_URL = "https://huggingface.co/datasets/argmaxinc/earnings22-kws-golden/resolve/main/data"

FILES = [
    "test-00000-of-00001.parquet",
    "validation-00000-of-00001.parquet",
]


def download_file(filename: str, output_dir: str = ".") -> None:
    """Download a file from the Hugging Face dataset."""
    url = f"{BASE_URL}/{filename}"
    output_path = os.path.join(output_dir, filename)

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"  Saved to {output_path}")


def main():
    for filename in FILES:
        download_file(filename)
    print("Done!")


if __name__ == "__main__":
    main()
