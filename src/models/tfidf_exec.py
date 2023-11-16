from src.models.tfidf import TFIDF, add_tfidf_to_sample
import json
from src.utils.bert_utils import evalute_list_sample_recall, evaluate_list_sample_precision, evalute_list_sample_f2

N_NEIGHBORS = 10 # @param {type:"integer"}
MAX_DF = 1 # @param {type:"number"}
MIN_DF = 1 # @param {type:"integer"}
NGRAM_MAX = 2 # @param {type:"integer"}
MAX_FEATURES = 0 # @param {type:"integer"}

tfidf_params = {
    'min_df': 1 if MIN_DF < 0 else MIN_DF,
    'max_df': 1 if MAX_DF < 0 else MAX_DF,
    'ngram_range': (1, NGRAM_MAX),
    'max_features': None if MAX_FEATURES == 0 else MAX_FEATURES
}

nn_params = {
    'n_neighbors': N_NEIGHBORS,
    'metric': 'cosine'
}

LAW_CORPUS_PATH = "./data/stage_2_law_corpus.json"

with open(LAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
    law_corpus = json.load(f)

tfidf = TFIDF(tfidf_params, nn_params)
tfidf.fit(law_corpus)

BM25_DATA_PATH = "./data/added_BM25_data.json"
BM25_TEST_PATH = "./data/added_BM25_test.json"

OUTPUT_TFIDF_DATA_PATH = "./data/added_tfidf_data.json"
OUTPUT_TFIDF_TEST_PATH = "./data/added_tfidf_test.json"

with open(BM25_DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Execute on train data")
add_tfidf_to_sample(
    data,
    tfidf,
    set_prediction= True,
    output_file_path= OUTPUT_TFIDF_DATA_PATH
)

print('Recall score: ')
print(evalute_list_sample_recall(data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(data))

with open(BM25_TEST_PATH, 'r', encoding='utf-8') as f:
  test_data = json.load(f)

print("Execute on test data")
add_tfidf_to_sample(
    test_data,
    tfidf,
    set_prediction= True,
    output_file_path= OUTPUT_TFIDF_TEST_PATH
)

print('Recall score: ')
print(evalute_list_sample_recall(test_data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(test_data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(test_data))