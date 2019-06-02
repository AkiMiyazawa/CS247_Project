# baselines.py
# 
# CS 247 Project @ UCLA

# Code adapted from 
# TODO: 
# (anton)
# Inser link

from collections import Counter
import pke
from main import get_raw_data


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
    return "{0:.2f}".format(f1)

def compare(model, samples=20):
  # Get sentences and keys from subset of our data
  sent, keys = get_raw_data()
  scores = []

  for i in range(samples):
    # Init pke model
    constructor = getattr(pke.unsupervised, model)
    extractor = constructor()
    # extractor = pke.unsupervised.SingleRank()
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
    f1 = f1_score(baseline, keys[i])
    scores.append(str(f1))
  return scores

if __name__ == '__main__':
  for model in ['TopicRank', 'SingleRank']:
    print("=========================================")
    scores = compare(model)
    print("F1 scores for {} are\n\n {}".format(model, ' '.join(scores)))
  print("=========================================")