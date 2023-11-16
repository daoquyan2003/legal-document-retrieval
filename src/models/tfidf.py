from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, List
from tqdm import tqdm
import json
import numpy as np

class TFIDF:
  def __init__(self, tfidf_params: Dict[str, Any] = {}, nn_params: Dict[str, Any] = {}):
    self.vectorizer = TfidfVectorizer(**tfidf_params)
    self.nn = NearestNeighbors(**nn_params)

  def fit(self, law_data: List[Dict[str, Any]]):
    self.articles = law_data
    law_corpus = [s['segment_text'] for s in law_data]
    self.weights = self.vectorizer.fit_transform(law_corpus)
    self.features = self.vectorizer.get_feature_names_out()

    self.nn.fit(self.weights)

  def __call__(self, sample):
    def top_features(idx, weights, features, k= 5):
      vec = weights.toarray()[idx,:]
      top_idx = np.argsort(vec)[::-1][:k]
      return [features[i] for i in top_idx]
    query = self.vectorizer.transform(sample)[0]
    dist, a_idx = self.nn.kneighbors(query)
    article_retrieved = [{'dist': d, 'article': self.articles[i], 'keywords': top_features(i, self.weights, self.features)} for d, i in [(dist[0][i], a_idx[0][i]) for i in range(len(dist[0]))]]
    return {
        'query': sample,
        'output': article_retrieved,
    }

def add_tfidf_to_sample(
    list_data,
    _model,
    set_prediction= True,
    output_file_path= None
):
    n_empty_relevant = 0
    for sample in tqdm(list_data, desc= 'Predicting...'):
        try:

            output = _model([sample['segment_text']])
            scores_dict = {
                article['article']['id']: article['dist'] for article in output['output']
            }

            sorted_scores = dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse= True))
            sample['tfidf_candidate'] = list(sorted_scores.keys())
            sample['tfidf_score'] = sorted_scores
            sample['predict_articles'] = sample['tfidf_candidate'] if set_prediction else None

        except Exception as e:
            print(e)
            print(sample["question_id"])
            n_empty_relevant += 1

    print("n_empty relevant: ", n_empty_relevant)

    json.dump(
        list_data, open(output_file_path, "w", encoding="utf-8"), ensure_ascii=False
    ) if output_file_path is not None else None
    return list_data
