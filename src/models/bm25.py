import numpy as np
from rank_bm25 import BM25Okapi
import json
from src.utils.bert_utils import evalute_list_sample_recall, evaluate_list_sample_precision, evalute_list_sample_f2

data_text_field = "text"

corpus_text_field = "text"

LAW_CORPUS_PATH = "./data/stage_2_law_corpus.json"

DATA_PATH = "./data/stage_2_data.json"

def find_article_position(given_statute_corpus, given_article_id):
    """
    Tìm vị trí của văn bản dựa trên ID của văn bản đó
    """
    for pos, article in enumerate(given_statute_corpus):
        if article.get("id") == given_article_id:
            return pos
    return None

def add_bm25_to_sample(
    list_data,
    output_file_path,
    _statute_corpus,
    _bm25_model,
    top_n,
    query_field="query",
    dump_data=False
):
    """
    Thực hiện infer BM25 và ghi kết quả infer vào dữ liệu
    """
    # list_f2_score = []
    n_empty_relevant = 0
    for sample in list_data:
        try:
            list_bm25_score = _bm25_model.get_scores(
                sample.get(data_text_field).split(" ")
            ).tolist()
            list_article_pos = np.argsort(list_bm25_score)[::-1][:top_n].tolist()
            sample["bm25_candidate"] = [
                _statute_corpus[pos].get("id") for pos in list_article_pos
            ]
            sample["bm25_score"] = {
                _statute_corpus[pos].get("id"): list_bm25_score[pos]
                for pos in list_article_pos
            }
            for article_id in sample.get("relevant_articles"):
                sample["bm25_score"][article_id] = list_bm25_score[
                    find_article_position(_statute_corpus, article_id)
                ]

            sample["predict_articles"] = sample["bm25_candidate"]
            # list_f2_score.append(evaluate_sample_recall(sample))
            # sample.pop("predict_articles")
        except Exception as e:
            print(e)
            print(sample["question_id"])
            n_empty_relevant += 1

    # print(f"Evaluate Bm25's recall score")
    print("n_empty relevant: ", n_empty_relevant)
    # if len(list_f2_score) > 0:
        # print(sum(list_f2_score) / len(list_f2_score))
    # else:
        # print("Maybe it is a test file, so that it does not have relevant article")
    if (dump_data is not False):
        json.dump(
            list_data, open(output_file_path, "w", encoding="utf-8"), ensure_ascii=False
        )
    return list_data

# Dựng mô hình BM25

with open(LAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
    law_corpus = json.load(f)

tokenized_corpus = [
    corpus.get(corpus_text_field).split(" ") for corpus in law_corpus
]

bm25_model = BM25Okapi(tokenized_corpus)

TOP_N = 100

OUTPUT_FILE_PATH = "./data/added_BM25_data.json"

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Execute on train data")
data = add_bm25_to_sample(
    data,
    OUTPUT_FILE_PATH,
    law_corpus,
    bm25_model,
    top_n=TOP_N,
    dump_data=True
)

print('Recall score: ')
print(evalute_list_sample_recall(data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(data))

# TOP_N = 2 gives the best F2 score
