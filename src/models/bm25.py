import numpy as np
from rank_bm25 import BM25Okapi
import json

data_text_field = "text"

corpus_text_field = "text"

LAW_CORPUS_PATH = "./data/stage_2_law_corpus.json"

DATA_PATH = "./data/stage_2_data.json"

TEST_PATH = "./data/stage_2_test.json"

def evaluate_sample_recall(given_sample):
    """
    Đánh giá điểm recall-score của một sample
    (dựa vào 2 trường: predict_article và relevant_article)
    """
    assert given_sample["predict_articles"] is not None, "Missing predict_articles"
    assert given_sample["relevant_articles"] is not None, "Missing relevant_articles"
    assert (
        len(given_sample["relevant_articles"]) > 0
    ), "number of relevant articles is 0"
    list_relevant_article = given_sample["relevant_articles"]
    list_predicted_article = given_sample["predict_articles"]
    num_relevant = len(list_relevant_article)
    num_predicted = len(list_predicted_article)
    num_true_positive = len(
        [
            _relevant_article
            for _relevant_article in list_relevant_article
            if _relevant_article in given_sample.get("predict_articles")
        ]
    )
    return num_true_positive / num_relevant

def evalute_list_sample_recall(list_sample):
    """
    Đánh giá điểm recall-score của một tập các sample
    (dựa vào 2 trường: predict_article và relevant_article)
    """
    list_recall_score = []
    for sample in list_sample:
        recall_i = evaluate_sample_recall(sample)
        list_recall_score.append(recall_i)
    return sum(list_recall_score) / (len(list_recall_score) + 1e-10)

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

    json.dump(
        list_data, open(output_file_path, "w", encoding="utf-8"), ensure_ascii=False
    )

def evaluate_sample_precision(given_sample):
    """
    Đánh giá điểm precision-score của một sample
    (dựa vào 2 trường: predict_article và relevant_article)
    """
    assert given_sample["predict_articles"] is not None, "Missing predict_articles"
    assert given_sample["relevant_articles"] is not None, "Missing relevant_articles"
    assert (
        len(given_sample["relevant_articles"]) > 0
    ), "number of relevant articles is 0"
    list_relevant_article = given_sample["relevant_articles"]
    list_predicted_article = given_sample["predict_articles"]
    num_relevant = len(list_relevant_article)
    num_predicted = len(list_predicted_article)
    num_true_positive = len(
        [
            _relevant_article
            for _relevant_article in list_relevant_article
            if _relevant_article in given_sample.get("predict_articles")
        ]
    )
    return num_true_positive / num_predicted

def evaluate_list_sample_precision(list_sample):
    list_precision_score = []
    for sample in list_sample:
        precision_i = evaluate_sample_precision(sample)
        list_precision_score.append(precision_i)
    return sum(list_precision_score) / (len(list_precision_score) + 1e-10)


def evaluate_sample_f2(given_sample):
    """
    Đánh giá điểm precision-score của một sample
    (dựa vào 2 trường: predict_article và relevant_article)
    """
    recall = evaluate_sample_recall(given_sample)
    precision = evaluate_sample_precision(given_sample)
    if precision == 0 and recall == 0:
        return 0.0
    return 5 * (precision * recall) / ((4 * precision) + recall)

def evalute_list_sample_f2(list_sample):
    """
    Đánh giá điểm recall-score của một tập các sample
    (dựa vào 2 trường: predict_article và relevant_article)
    """
    # YOUR CODE HERE
    list_f2_score = []
    for sample in list_sample:
        f2_i = evaluate_sample_f2(sample)
        list_f2_score.append(f2_i)
    return sum(list_f2_score) / (len(list_f2_score) + 1e-10)

from rank_bm25 import BM25Okapi

# Dựng mô hình BM25

with open(LAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
    law_corpus = json.load(f)

tokenized_corpus = [
    corpus.get(corpus_text_field).split(" ") for corpus in law_corpus
]

bm25_model = BM25Okapi(tokenized_corpus)

TOP_N = 2 # @param {type:"integer"}

OUTPUT_FILE_PATH = "./data/added_BM25_data.json"

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Execute on train data")
add_bm25_to_sample(
    data,
    OUTPUT_FILE_PATH,
    law_corpus,
    bm25_model,
    top_n=TOP_N
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
