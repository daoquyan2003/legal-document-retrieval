running_name = "single-article-BERT"
ESP = 1e-10

import torch
from torch.nn.functional import softmax
import pandas as pd
import json

def min_max_normalize_list(a):
    return [(e - min(a)) / (max(a) - min(a) + ESP) for e in a]

def save_csv_predict_result(
    list_qid,
    list_article_identity,
    list_label,
    list_bm25_score=None,
    list_bert_score=None,
):
    csv_dict = {
        "qid": [],
        "article_id": [],
        "label": [],
        "bm25_score": [],
        "bert_score": [],
    }

    def add_row(_qid, _article_id, _label, _bm25_score=None, _bert_score=None):
        csv_dict["qid"].append(_qid)
        csv_dict["article_id"].append(_article_id)
        csv_dict["label"].append(_label)
        csv_dict["bm25_score"].append(_bm25_score)
        csv_dict["bert_score"].append(_bert_score)

    for i in range(len(list_qid)):
        qid = list_qid[i]
        for j in range(len(list_label[i])):
            label = list_label[i][j]
            article_id = list_article_identity[i][j]
            bm25_score = list_bm25_score[i][j] if list_bm25_score is not None else None
            bert_score = list_bert_score[i][j] if list_bert_score is not None else None
            add_row(qid, article_id, label, bm25_score, bert_score)

    pd.DataFrame(csv_dict).to_csv("./single-article-BERT_predict_result.csv", index=False)


def synthetic_step_output_v2(outputs):
    def send_data_to_the_right_device(_data):
        return _data.item() if torch.is_tensor(_data) else _data

    save_predict_dict = dict()
    save_label_dict = dict()
    save_bm25_score_dict = dict()
    save_article_identity_dict = dict()

    # Xét qua tất cả pair trong outputs và tổng hợp lại theo qid
    for epoch_output in outputs:
        list_bm25_score = epoch_output.get("bm25_score")
        list_model_predict = epoch_output.get("model_predict")
        list_label = epoch_output.get("label")
        list_qid = epoch_output.get("list_qid")
        list_article_identity = epoch_output.get("list_article_identity")
        for i, qid in enumerate(list_qid):
            qid = send_data_to_the_right_device(qid)
            # model_predict = list_model_predict[i].tolist()
            # assert len(model_predict) == 2, 'Model predict has inappropriate shape'
            model_predict_prob = send_data_to_the_right_device(
                softmax(list_model_predict[i], dim=0)[1]
            )
            label = send_data_to_the_right_device(list_label[i])
            bm25_score = send_data_to_the_right_device(list_bm25_score[i])
            article_identity = send_data_to_the_right_device(list_article_identity[i])

            if qid in save_predict_dict.keys():
                save_predict_dict[qid].append(model_predict_prob)
            else:
                save_predict_dict[qid] = [model_predict_prob]

            if qid in save_label_dict.keys():
                save_label_dict[qid].append(label)
            else:
                save_label_dict[qid] = [label]

            if qid in save_bm25_score_dict.keys():
                save_bm25_score_dict[qid].append(bm25_score)
            else:
                save_bm25_score_dict[qid] = [bm25_score]

            if qid in save_article_identity_dict.keys():
                save_article_identity_dict[qid].append(article_identity)
            else:
                save_article_identity_dict[qid] = [article_identity]

    save_bm25_predict_list = []
    save_bert_predict_list = []
    save_label_list = []
    save_article_identity_list = []
    list_qid = []

    for qid in save_predict_dict.keys():
        list_qid.append(qid)
        bert_score = save_predict_dict.get(qid)
        bm25_score = save_bm25_score_dict.get(qid)
        bm25_score_norm = min_max_normalize_list(bm25_score)
        bert_score_norm = min_max_normalize_list(bert_score)
        save_bm25_predict_list.append(bm25_score)
        save_bert_predict_list.append(bert_score)
        save_label_list.append(save_label_dict.get(qid))
        save_article_identity_list.append(save_article_identity_dict.get(qid))

    return (
        list_qid,
        save_article_identity_list,
        save_label_list,
        save_bm25_predict_list,
        save_bert_predict_list,
    )

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

def add_bert_to_sample(
    list_data,
    _output,
    set_prediction= True,
    output_file_path= None
):
    num_empty = 0
    for sample in list_data:
      try:
        q_id = sample['question_id']
        scores = []
        for q in _output:
          if q['list_qid'][0] == q_id and q['label'] == 1:
            article = q['list_article_identity'][0]
            prob = torch.max(torch.nn.functional.softmax(q['model_predict'].detach().cpu().float(), dim= -1)).item()
            scores.append((article, prob))
        articles = {k: v for k, v in scores}
        articles = dict(sorted(articles.items(), key=lambda item: item[1], reverse= True))
        sample['bert_candidate'] = list(articles.keys())
        sample['bert_prob'] = articles
      except Exception as e:
        print(e)
        print(q)
        print(sample['question_id'])
        num_empty += 1
    print(num_empty)
    json.dump(
        list_data, open(output_file_path, "w", encoding="utf-8"), ensure_ascii=False
    ) if output_file_path is not None else None