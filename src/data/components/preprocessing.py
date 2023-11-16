import json
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize

def rename_keys(d):
    for key in list(d.keys()):
        if key == "id":
            d["question_id"] = d.pop(key)

def rename_keys_in_list(list_of_dicts):
    for d in list_of_dicts:
        rename_keys(d)

def preprocess_data(dump_data=False):
    LAW_2023_JSON_PATH = "./data/ALQAC_2023_training_data_v1.1/law.json"
    QUESTION_2023_JSON_PATH = "./data/ALQAC_2023_training_data_v1.1/train.json"
    # TEST_JSON_PATH = "./data/ALQAC_2023_training_data_v1.1/public_test.json"

    LAW_2022_JSON_PATH = "./data/ALQAC_2023_training_data_v1.1/additional_data/ALQAC_2022_training_data/law.json"
    QUESTION_2022_JSON_PATH = "./data/ALQAC_2023_training_data_v1.1/additional_data/ALQAC_2022_training_data/question.json"

    LAW_ZALO_JSON_PATH = "./data/ALQAC_2023_training_data_v1.1/additional_data/zalo/zalo_corpus.json"
    QUESTION_ZALO_JSON_PATH = "./data/ALQAC_2023_training_data_v1.1/additional_data/zalo/zalo_question.json"

    with open(LAW_2023_JSON_PATH, 'r', encoding='utf-8') as f:
        law_corpus = json.load(f)

    with open(LAW_2022_JSON_PATH, 'r', encoding='utf-8') as f:
        law_corpus_2022 = json.load(f)

    with open(LAW_ZALO_JSON_PATH, 'r', encoding='utf-8') as f:
        law_corpus_zalo = json.load(f)

    for item in law_corpus_2022:
        law_corpus.append(item)

    for item in law_corpus_zalo:
        law_corpus.append(item)

    with open(QUESTION_2023_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(QUESTION_2022_JSON_PATH, 'r', encoding='utf-8') as f:
        data_2022 = json.load(f)

    with open(QUESTION_ZALO_JSON_PATH, 'r', encoding='utf-8') as f:
        data_zalo = json.load(f)

    rename_keys_in_list(data_2022)
    rename_keys_in_list(data_zalo)

    for item in data_2022:
        data.append(item)

    for item in data_zalo:
        data.append(item)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    def combine_new_article_id(law_id, article_id):
        return f"{law_id}|{article_id}"

    flat_law_corpus = [
        {
            **dieu_luat,
            "id": combine_new_article_id(bo_luat.get('id'),dieu_luat.get('id'))
        }
        for bo_luat in law_corpus for dieu_luat in bo_luat.get('articles')
    ]

    update_article_data = [
        {
            **sample,
            "relevant_articles": [
                combine_new_article_id(article.get('law_id'), article.get('article_id'))
                for article in sample.get("relevant_articles")
            ]
        } for sample in train_data
    ]

    update_article_test_data = [
        {
            **sample,
            "relevant_articles": [
                combine_new_article_id(article.get('law_id'), article.get('article_id'))
                for article in sample.get("relevant_articles")
            ]
        } for sample in test_data
    ]

    stage_2_law_corpus = [
        {
            **dieu_luat,
            "segment_text": word_tokenize(dieu_luat.get('text'), format="text")
        } for dieu_luat in flat_law_corpus
    ]

    stage_2_data = [
        {
            **sample,
            "segment_text": word_tokenize(sample.get('text'), format="text")
        }
        for sample in update_article_data
    ]

    stage_2_test_data = [
        {
            **sample,
            "segment_text": word_tokenize(sample.get('text'), format="text")
        }
        for sample in update_article_test_data
    ]

    if dump_data is not False:
        STAGE_2_LAW_CORPUS_PATH = './data/stage_2_law_corpus.json'
        STAGE_2_DATA_PATH = './data/stage_2_data.json'
        STAGE_2_TEST_PATH = './data/stage_2_test.json'


        with open(STAGE_2_LAW_CORPUS_PATH, 'w', encoding='utf-8') as f:
            json.dump(stage_2_law_corpus, f, ensure_ascii=False)   

        with open(STAGE_2_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(stage_2_data, f, ensure_ascii=False)

        with open(STAGE_2_TEST_PATH, 'w', encoding='utf-8') as f:
            json.dump(stage_2_test_data, f, ensure_ascii=False)

    return stage_2_law_corpus, stage_2_data, stage_2_test_data


stage_2_law_corpus, stage_2_data, stage_2_test_data = preprocess_data(dump_data=True)
