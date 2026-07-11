import json

with open("data/processed/train.jsonl", "r", encoding="utf-8") as f:
    first_line = f.readline()

print(json.loads(first_line))