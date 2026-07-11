import pandas as pd
import json

# Load the CSV
df = pd.read_csv("data/raw/Nepali_news_dataset.csv", encoding="utf-8")

# Keep only the columns we need
df = df[["text", "title"]].dropna()

# Create JSONL file
with open("data/processed/train.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        example = {
            "instruction": "Summarize the following Nepali news article.",
            "input": str(row["text"]),
            "output": str(row["title"])
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print("✅ Dataset created successfully!")
print("Total samples:", len(df))