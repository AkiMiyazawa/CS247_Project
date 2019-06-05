# baselines.py
# 
# CS 247 Project @ UCLA

# Code adapted from 
# TODO: 
# (anton)
# Insert link

from collections import Counter
import pke
from main import get_raw_data
from nltk.corpus import stopwords

# define a list of stopwords
stoplist = stopwords.words('english')

# TODO:
# (anton)
# add import instead of duplicating code once the branch is merged
def f1_score(prediction, ground_truth):
    # both prediction and grount_truth should be list of words
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compare(model, samples=10):
  # Get sentences and keys from subset of our data
  sent, keys = get_raw_data()
  scores = []

  # Init pke model
 
  for i in range(100, 100+samples):
    constructor = getattr(pke.unsupervised, model)
    extractor = constructor()
    # load the content of the document
    extractor.load_document(input=sent[i], language='en')

    # keyphrase candidate selection
    extractor.candidate_selection()

    # candidate weighting
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 5 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=5)
    baseline = list(map(lambda x : x[0], keyphrases))


    base = []
    for j in baseline:
      base += j.split()
    # print(base)
    f1 = f1_score(base, keys[i])
    # print(f1)
    scores.append(f1)
    if i % 10 == 0:
      print("Calculating...")
  return scores


if __name__ == '__main__':
  for model in ['TopicRank', 'SingleRank', 'TextRank', 'MultipartiteRank', 'PositionRank']:
    print("=========================================")
    scores = compare(model,samples=30)
    avg = "{0:.4f}".format(sum(scores) / len(scores))
    print('Average F1 score is {}'.format( avg ))
  print("=========================================")