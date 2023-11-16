from typing import Dict

class Task3ContextUtility:
    def __init__(self, statute_corpus: Dict, data_text_field: str = "text", corpus_text_field: str = "text"):
        self.statute_corpus = statute_corpus
        self.data_text_field = data_text_field
        self.corpus_text_field = corpus_text_field

    def create_gen_sample(self, list_sample, is_include_gold, n_candidate):
        list_train_gen_sample = []
        article_content_field = self.corpus_text_field

        query_content_field = self.data_text_field

        def get_article_content_by_aid(article_id):
            for article in self.statute_corpus:
                if article.get("article_id") == article_id:
                    return article.get(article_content_field)
            return None

        for sample in list_sample:
            qid = sample.get("question_id")
            query = sample.get(query_content_field)
            list_candidate = [
                article_id for article_id in sample.get("bm25_candidate")[:n_candidate]
            ]
            list_relevant_article_id = None
            if is_include_gold:
                list_relevant_article_id = sample.get("relevant_articles")
                list_candidate += list_relevant_article_id
            list_candidate = set(list_candidate)

            for article_id in list_candidate:
                retrieval_label = None
                yn_label = None
                if is_include_gold:
                    retrieval_label = int(article_id in list_relevant_article_id)
                    yn_label = (
                        int(sample.get("yn_label") == "Y")
                        if retrieval_label == 1
                        else 2
                    )

                list_train_gen_sample.append(
                    {
                        "q": query,
                        "a": get_article_content_by_aid(article_id),
                        "qid": qid,
                        "aid": article_id,
                        "bm25_score": sample.get("bm25_score")[article_id],
                        "yn_label": yn_label,
                        "label": retrieval_label,
                    }
                )
        return list_train_gen_sample