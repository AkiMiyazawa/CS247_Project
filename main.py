from nltk.tokenize import word_tokenize
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import csv
from summa import keywords as kw

################################################################
# DataLoader works fine
# Edits may have errors due to <SP> and <TAB> consistency
# My computer broke, so I couldn't test if EmbeddingLayer works
################################################################

class DataLoader:
    def __init__(self):
        self.tokenized_X = None
        self.tokenized_y = None
        self.X = None
        self.y = None
        self.n_vocab = 0
        self.vocab = {}
        self.vocab_inv = {}
        pass        

    def load_data(self, sentences=None, keywords=None, verbose=False, load_one_hot=False):
        if sentences is None or keywords is None:
            sentences = ['toy example', 'foo#', 'bar baz$ C++!', 'todo']
            keywords = [['toy', 'easy'], ['foo'], ['C++', 'Java', 'Lists'], ['test']]
        
        print('tokenizing...')
        self.tokenized_X = self._tokenize(dataset=sentences)
        self.tokenized_y = self._tokenize(dataset=keywords, keywords=True)
        print('constructing vocab...')
        self._construct_vocabulary(self.tokenized_X, self.tokenized_y)
        print('constructing one-hot...')
        if load_one_hot:
            onehot_X = self._one_hot(dataset=self.tokenized_X)
            onehot_y = self._one_hot(dataset=self.tokenized_y, keywords=True)
            self.X = onehot_X #torch.tensor(onehot_X)
            self.y = onehot_y #torch.tensor(onehot_y)
        else:
            self.X = self.tokenized_X #torch.tensor(onehot_X)
            self.y = self.tokenized_y #torch.tensor(onehot_y)
        if verbose:
            print('====================tokenized====================')
            print(self.tokenized_X)
            print(self.tokenized_y)
            '''
            print('====================onehot====================')
            print(onehot_X)
            print(onehot_y)
            print('====================pytorch====================')
            print(self.X)
            print(self.y)
            '''
        print('data loaded')

    def get_data(self):
        return self.X, self.y

    def get_metadata(self):
        return self.vocab, self.vocab_inv, self.n_vocab 

    def _tokenize(self, dataset=None, keywords=False):
        tokenized_dataset = []
        # iterate through all samples in dataset
        for sample in dataset:
            tokenized_sample = []
            # iterate through all sentences in sample
            if keywords:
                tokenized_sample = []
                for keyword in sample:
                    tokenized_keyword = word_tokenize(keyword)
                    if len(tokenized_keyword) > 1:
                        if tokenized_keyword[1] == '#':
                            tokenized_keyword = [tokenized_keyword[0] + tokenized_keyword[1]]
                        else:
                            print('WARNING: split keyword {}'.format(','.join(tokenized_keyword)))
                    tokenized_sample.append(tokenized_keyword[0])
            else:
                # tokenize the entire sentence
                tokenized_sample = word_tokenize(sample)
                i_offset = 0
                for i, t in enumerate(tokenized_sample):
                    i -= i_offset
                    if t == '#' and i > 0:
                        left = tokenized_sample[:i-1]
                        joined = [tokenized_sample[i - 1] + t]
                        right = tokenized_sample[i + 1:]
                        tokenized_sample = left + joined + right
                        i_offset += 1

            tokenized_dataset.append(tokenized_sample)
        return tokenized_dataset # Out[]: list of list of tokens 

    def _one_hot(self, dataset=None, keywords=False):
        # find the maximum length input:
        n_dataset = len(dataset)

        # determine the maximum length
        n_sample = 0
        for sample in dataset:
            if len(sample) > n_sample:
                n_sample = len(sample) 
        n_sample += 2 # for <BOS> and <EOS> tokens
        Z = np.zeros([n_dataset, n_sample, self.n_vocab])

        # one hot encode the dataset
        for i, sample in enumerate(dataset):
            if keywords:
                for j, word in enumerate(sample):
                    Z[i, j, self.vocab[word]] = 1
            else:
                Z[i, 0, 1] = 1 # <BOS> token
                for j, word in enumerate(sample):
                    Z[i, j+1, self.vocab[word]] = 1
                Z[i, j+2, 2] = 1 # <EOS> token
                Z[i, j+3:n_sample, 0] = 1 # <PAD> token
        return Z

    def _construct_vocabulary(self, dataset_X, dataset_y):
        # get all samples
        all_samples = dataset_X+dataset_y
        num_spec_tokens = 3 # for <BOS> <EOS> and <PAD>
        self.vocab = {'<PAD>':0, '<BOS>':1, '<EOS>':2}
        self.vocab_inv = {0:'<PAD>', 1:'<BOS>', 2:'<EOS>'}
        self.n_vocab = num_spec_tokens
        for sample in all_samples:
            for word in sample:
                if word not in self.vocab.keys():
                    self.vocab[word] = self.n_vocab
                    self.vocab_inv[self.n_vocab] = word
                    self.n_vocab += 1

# source: https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
# TODO: check if this implementation is correct
class EmbeddingLayer:
    def __init__(self, n_vocab, embedding_dim, window_size, vocab):
        self.n_vocab = n_vocab
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.vocab = vocab
    
    def gen_embeddings(self, dataset, epochs, lr):
        print('generating words')
        self._init_embeddings()
        print('debug0')
        self._gen_idx_pairs(dataset)
        print('debug1')
        self._train_embeddings(epochs, lr)
        print('debug2')
    
    def get_embeddings(self, dataset, load_one_hot=False):
        # TODO: concatenate the pytorch vectors here!
        num_sentences, num_words, _ = dataset.shape
        out = np.zeros([num_sentences, num_words, self.embedding_dim])
        W = self.W1.detach().numpy()
        for i, sentence in enumerate(dataset):
            indices = [self.vocab[word] for word in sentence]
            Z = np.hstack([W[:,idx] for idx in indices]).T
            out[i] = Z
        return out
    
    def _gen_idx_pairs(self, dataset, load_one_hot=False):
        idx_pairs = []
        for sentence in dataset:
            indices = [self.vocab[word] for word in sentence]
            for center_word_pos in range(len(indices)):
                for w in range(-self.window_size, self.window_size + 1):
                    context_word_pos = center_word_pos + w
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))
        self.idx_pairs = idx_pairs
    
    def _init_embeddings(self):
        self.W1 = Variable(torch.randn(self.embedding_dim, self.n_vocab).float(), requires_grad=True)
        self.W2 = Variable(torch.randn(self.n_vocab, self.embedding_dim).float(), requires_grad=True)
    
    def _train_embeddings(self, epochs, lr):
        for epoch in range(epochs):
            loss_val = 0
            for data, target in self.idx_pairs:
                x = torch.zeros(self.n_vocab).float()
                x[data] = 1.0
                y_true = Variable(torch.from_numpy(np.array([target])).long())
                z1 = torch.matmul(self.W1, x)
                z2 = torch.matmul(self.W2, z1)
                log_softmax = F.log_softmax(z2, dim=0)
                '''
                print('data')
                print(data)
                print('target')
                print(target)
                print('y_true')
                print(y_true)
                print('log_softmax')
                print(log_softmax)
                '''
                loss = F.nll_loss(log_softmax.view(1,-1), y_true)
                loss_val += loss.data.item()
                loss.backward()
                self.W1.data -= lr * self.W1.grad.data
                self.W2.data -= lr * self.W2.grad.data

                self.W1.grad.data = self.W1.grad.data.zero_()
                self.W2.grad.data = self.W2.grad.data.zero_()
            if epoch % 10 == 0:    
                print('Loss at epoch {}: {}'.format(epoch,loss_val/len(self.idx_pairs)))

def get_raw_data():
    # load the raw data
    fp = open('data_subset.csv')
    csv_reader = csv.reader(fp, delimiter=',')
    sentences = []
    keywords = []
    for row in csv_reader:
        [sentence, keyword, _] = row
        sentences.append(sentence)
        keywords.append(keyword.split(';'))
    return sentences[1:], keywords[1:]

def main():
    # hyper parameters
    embedding_dim = 10
    window_size = 3
    epochs = 50
    lr = 1e-3
    model = 'textrank'

    corpus, keywords = get_raw_data()
    assert len(corpus) == len(keywords)

    ######### for debugging #########
    max_len = 2
    assert max_len <= len(corpus)
    corpus = corpus[:max_len]
    keywords = keywords[:max_len]
    #################################

    if model == 'Our':
        data_loader = DataLoader()
        data_loader.load_data(corpus, keywords)
        X, y = data_loader.get_data()    
        vocab, inv_vocab, n_vocab = data_loader.get_metadata()
        embedder = EmbeddingLayer(n_vocab, embedding_dim, window_size, vocab)
        embedder.gen_embeddings(dataset=X, epochs=epochs, lr=lr)
        X_emb = embedder.get_embeddings(X)
        y_emb = embedder.get_embeddings(X)
    elif model == 'textrank':
        print('number of samples: ' + str(len(corpus)))
        predictions = []
        for i, sentence in enumerate(corpus):
            if i % 100 == 0:
                print('evaluating sample: ' + str(i))
            prediction = kw.keywords(sentence).split('\n')
            predictions.append(prediction)
        print('done!')

    #################### This is from fork ####################
    from collections import Counter
    def f1_score(prediction, ground_truth):
        # both prediction and grount_truth should be list of words
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall
    ###########################################################

    assert len(predictions) == len(keywords)
    f1 = 0
    precision = 0
    recall = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        keyword = keywords[i]
        if i%100 == 0:
            print('-------------------------')
            print('prediction:')
            print(prediction)
            print('ground-truth:')
            print(keyword)
        f1_sample, precision_sample, recall_sample = f1_score(prediction, keyword)
        f1 += f1_sample
        precision += precision_sample
        recall += recall_sample

    f1 /= len(predictions)
    precision /= len(predictions)
    recall /= len(predictions)
    print('F1:{}'.format(f1))
    print('precision:{}'.format(precision))
    print('recall:{}'.format(recall))

if __name__ == '__main__':
    main()
