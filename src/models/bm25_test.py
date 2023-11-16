from src.models.bm25 import add_bm25_to_sample
from src.utils.bert_utils import evalute_list_sample_recall, evaluate_list_sample_precision, evalute_list_sample_f2
import json

from rank_bm25 import BM25Okapi

data_text_field = "text"

corpus_text_field = "text"

LAW_CORPUS_PATH = "./data/stage_2_law_corpus.json"

with open(LAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
  law_corpus = json.load(f)

tokenized_corpus = [
    corpus.get(corpus_text_field).split(" ") for corpus in law_corpus
]

bm25_model = BM25Okapi(tokenized_corpus)

TEST_PATH = "./data/stage_2_test.json"
TEST_OUTPUT_FILE_PATH = "./data/added_BM25_test.json"

with open(TEST_PATH, 'r', encoding='utf-8') as f:
  test_data = json.load(f)

TOP_N = 100

print("Execute on test data")
test_data = add_bm25_to_sample(
    test_data,
    TEST_OUTPUT_FILE_PATH,
    law_corpus,
    bm25_model,
    top_n=TOP_N,
    dump_data=True
)

print('Recall score: ')
print(evalute_list_sample_recall(test_data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(test_data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(test_data))