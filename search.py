import json
import time
import datasets
import requests

from tqdm import tqdm
from googlesearch import search
from duckduckgo_search import DDGS


# Function to parse JSONL file and extract required information
def parse_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in tqdm(file, desc="Parsing JSONL file"):
            entry = json.loads(line)
            custom_id = entry["custom_id"]
            word = custom_id.split("_")[1]
            try:
                search_keywords = json.loads(
                    entry["response"]["body"]["choices"][0]["message"]["content"]
                )["search_keyword"]
            except KeyError:
                search_keywords = json.loads(
                    entry["response"]["body"]["choices"][0]["message"]["content"]
                )["search_keywords"]
            data.append((word, search_keywords))
    return data


def get_search_results(search_keyword):
    results = []
    while True:
        try:
            for result in search(
                search_keyword, unique=True, advanced=True, num_results=20
            ):
                results.append(
                    {"title": result.title, "description": result.description}
                )
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(
                    "Received 429 Too Many Requests error. Waiting for 5 seconds before retrying..."
                )
                time.sleep(5)
            else:
                raise e
    return results


def get_search_results_ddgs(search_keyword):
    results = []
    while True:
        try:
            for result in DDGS().text(search_keyword, region="id-id", max_results=20):
                results.append(
                    {"title": result["title"], "description": result["body"]}
                )
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(
                    "Received 429 Too Many Requests error. Waiting for 5 seconds before retrying..."
                )
                time.sleep(5)
            else:
                raise e
    return results


# Function to build Hugging Face dataset
def build_dataset(data):
    dataset = []
    for word, search_keywords in tqdm(data, desc="Building dataset"):
        for keyword in search_keywords:
            # time.sleep(1)
            # search_results = get_search_results(keyword)
            search_results = get_search_results_ddgs(keyword)
            dataset.append(
                {
                    "word": word,
                    "search_keyword": keyword,
                    "search_results": search_results,
                }
            )
    return datasets.Dataset.from_dict(dataset)


# Main function
def main():
    file_path = "/home/haznitrama/scale-resources/batch_output_4.jsonl"
    data = parse_jsonl(file_path)
    dataset = build_dataset(data)
    dataset.save_to_disk("/home/haznitrama/scale-resources/gpt_search_results_dataset")


if __name__ == "__main__":
    main()
