running_name = "single-article-BERT"
ESP = 1e-10

import torch
from torch.nn.functional import softmax
import pandas as pd

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

    pd.DataFrame(csv_dict).to_csv(running_name, index=False)


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