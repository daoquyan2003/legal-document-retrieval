import fasttext
import fasttext.util
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import json
from src.utils.bert_utils import evalute_list_sample_recall, evaluate_list_sample_precision, evalute_list_sample_f2

fasttext.util.download_model('vi', if_exists='ignore')

ft = fasttext.load_model('cc.vi.300.bin')

def add_w2v_to_sample(list_data, law_corpus, _model, top_k= 5, set_prediction= True, output_file_path= None):
  corpus_w2v = [_model.get_sentence_vector(l['segment_text']).tolist() for l in law_corpus]
  nn = NearestNeighbors(n_neighbors= top_k, metric= 'cosine')
  nn.fit(corpus_w2v)
  num_empty = 0
  def get_article(id, law_corpus):
    for l in law_corpus:
      if l['id'] == id:
        return l
  for sample in tqdm(list_data, desc= 'Predicting...'):
    try:
      q_emb = ft.get_sentence_vector(sample['segment_text']).tolist()
      dist, a_idx = nn.kneighbors([q_emb])
      article_retrieved = [{'dist': d, 'article': law_corpus[i]['id']} for d, i in [(dist[0][i], a_idx[0][i]) for i in range(len(dist[0]))]]
      scores_dict = {
        article['article']: article['dist'] for article in article_retrieved
      }
      sorted_scores = dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse= True))
      sample['w2v_candidate'] = list(sorted_scores.keys())
      sample['w2v_score'] = sorted_scores
      sample['predict_articles'] = sample['w2v_candidate'] if set_prediction else None
    except Exception as e:
      print(e)
      print(sample['question_id'])
      num_empty += 1
  print(num_empty)
  json.dump(
      list_data, open(output_file_path, "w", encoding="utf-8"), ensure_ascii=False
  ) if output_file_path is not None else None

LAW_CORPUS_PATH = "./data/stage_2_law_corpus.json"
with open(LAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
    law_corpus = json.load(f)

TFIDF_DATA_PATH = "./data/added_tfidf_data.json"
OUTPUT_W2V_DATA_PATH = './data/added_w2v_data.json'

with open(TFIDF_DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('Execute on train data')
add_w2v_to_sample(
    data,
    law_corpus,
    ft,
    top_k= 10,
    output_file_path= OUTPUT_W2V_DATA_PATH
)

print('Recall score: ')
print(evalute_list_sample_recall(data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(data))

TFIDF_TEST_PATH = "./data/added_tfidf_test.json"
OUTPUT_W2V_TEST_PATH = './data/added_w2v_test.json'

with open(TFIDF_TEST_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print('Execute on test data')
add_w2v_to_sample(
    test_data,
    law_corpus,
    ft,
    top_k= 10,
    output_file_path= OUTPUT_W2V_TEST_PATH
)

print('Recall score: ')
print(evalute_list_sample_recall(test_data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(test_data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(test_data))