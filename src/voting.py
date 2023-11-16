import json
from src.utils.bert_utils import evalute_list_sample_recall, evaluate_list_sample_precision, evalute_list_sample_f2

def vote_per_sample(sample):
  bert_candidate = sample['bert_candidate']
  w2v_candidate = sample['w2v_candidate']
  tfidf_candidate = sample['tfidf_candidate']
  bm25_candidate = sample['bm25_candidate']
  n = max([len(bert_candidate), len(bm25_candidate), len(tfidf_candidate), len(w2v_candidate)])
  candidates = {}

  def calc_votes(candidates, n, bert= False):
    if bert:
      return [(candidate, 1) for candidate in candidates]
    return [(candidate, n + 1 - i) for i, candidate in enumerate(candidates)]

  for c, v in calc_votes(bert_candidate, n, bert= True):
    if c not in candidates.keys():
      candidates[c] = 0 + v
    else:
      candidates[c] += v

  for c, v in calc_votes(w2v_candidate, n):
    if c not in candidates.keys():
      candidates[c] = 0 + v
    else:
      candidates[c] += v

  for c, v in calc_votes(tfidf_candidate, n):
    if c not in candidates.keys():
      candidates[c] = 0 + v
    else:
      candidates[c] += v

  for c, v in calc_votes(bm25_candidate, n):
    if c not in candidates.keys():
      candidates[c] = 0 + v
    else:
      candidates[c] += v
  candidates = dict(sorted(candidates.items(), key=lambda item: item[1], reverse= True))
  return candidates

def vote(list_data, top_n= 1):
  num_empty = 0
  for sample in list_data:
    try:
      candidates = vote_per_sample(sample)
      predicted = list(candidates.keys())[:top_n]
      sample['predict_articles'] = predicted
      n_vote = {k:v for k, v in [(c, candidates[c]) for c in predicted]}
      sample['votes'] = n_vote
    except Exception as e:
      print(e)
      print(sample['question_id'])
      num_empty += 1
  print(num_empty)

BERT_TEST_PATH = "./data/added_bert_test_data.json"
with open(BERT_TEST_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

vote(test_data, 2)

print('Recall score: ')
print(evalute_list_sample_recall(test_data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(test_data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(test_data))