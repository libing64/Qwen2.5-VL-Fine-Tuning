import numpy as np
import json


def read_jsonl_basic(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_data(image_dir, entry):
    SYSTEM_MESSAGE = """You are a Vision Language Model specialized in extracting structured data from visual representations of palette manifests.
Your task is to analyze the provided image of a palette manifest and extract the relevant information into a well-structured JSON format.
The palette manifest includes details such as item names, quantities, dimensions, weights, and other attributes.
Focus on identifying key data fields and ensuring the output adheres to the requested JSON structure.
Provide only the JSON output based on the extracted information. Avoid additional explanations or comments."""

    question = 'extract data in JSON format'
    answer = entry["suffix"]
    image_path = image_dir + "/" + entry["image"]

    message = {
        "id": entry["image"],
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + question
            },
            {
                "from": "gpt",
                "value": answer
            },
        ],
        "system": SYSTEM_MESSAGE,
    }
    return message


def convert2sharegpt(jsonl_file, dst, image_dir):

    output = []
    data = read_jsonl_basic(jsonl_file)

    for item in data:
        output.append(format_data(image_dir, item))

    with open(dst, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    root = "train/annotations.jsonl"
    dst = "manifest.json"
    image_dir = "/home/libing/dataset/pallet-load-manifest-json-2/train/"
    convert2sharegpt(root, dst, image_dir)
