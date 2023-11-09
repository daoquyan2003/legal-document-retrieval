from sklearn.model_selection import train_test_split
import json

DATA_PATH = "./data/added_BM25_test.json"

VALID_PATH = "./data/added_BM25_valid_data.json"

TEST_PATH = "./data/added_BM25_test_data.json"

with open(DATA_PATH, 'r', encoding='utf-8') as f:
  data = json.load(f)

valid_data, test_data = train_test_split(data, test_size=0.5, shuffle=True, random_state=42)

with open(TEST_PATH, 'w', encoding='utf-8') as f:
  json.dump(test_data, f, ensure_ascii=False)

with open(VALID_PATH, 'w', encoding='utf-8') as f:
  json.dump(valid_data, f, ensure_ascii=False)
    