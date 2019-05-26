from nltk.tokenize import word_tokenize
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import csv

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

    def load_data(self, sentences=None, keywords=None, verbose=False):
        if sentences is None or keywords is None:
            sentences = ['toy example', 'foo#', 'bar baz$ C++!', 'todo']
            keywords = [['toy', 'easy'], ['foo'], ['C++', 'Java', 'Lists'], ['test']]
        
        print('tokenizing...')
        self.tokenized_X = self._tokenize(dataset=sentences)
        self.tokenized_y = self._tokenize(dataset=keywords, keywords=True)
        print('constructing vocab...')
        self._construct_vocabulary(self.tokenized_X, self.tokenized_y)
        print('constructing one-hot...')
        onehot_X = self._one_hot(dataset=self.tokenized_X)
        onehot_y = self._one_hot(dataset=self.tokenized_y, keywords=True)
        self.X = onehot_X #torch.tensor(onehot_X)
        self.y = onehot_y #torch.tensor(onehot_y)
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
                        print('WARNING: split keyword {}'.format(','.join(tokenized_keyword)))
                    tokenized_sample.append(tokenized_keyword[0])
            else:
                # tokenize the entire sentence
                tokenized_sample = word_tokenize(sample)
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
    def __init__(self, n_vocab, embedding_dim, window_size):
        self.n_vocab = n_vocab
        self.embedding_dim = embedding_dim
        self.window_size = window_size
    
    def gen_embeddings(self, dataset, epochs, lr):
        self._init_embeddings()
        self._gen_idx_pairs(dataset)
        self._train_embeddings(epochs, lr)
    
    def get_embeddings(self, dataset):
        # TODO: concatenate the pytorch vectors here!
        num_sentences, num_words, _ = dataset.shape
        out = np.zeros([num_sentences, num_words, self.embedding_dim])
        W = self.W1.detach().numpy()
        for i, sentence in enumerate(dataset):
            indices = [np.where(word == 1)[0] for word in sentence]
            Z = np.hstack([W[:,idx] for idx in indices]).T
            out[i] = Z
        return out
    
    def _gen_idx_pairs(self, dataset):
        idx_pairs = []
        for sentence in dataset:
            indices = [np.where(word == 1) for word in sentence]
            for center_word_pos in range(len(indices)):
                for w in range(-self.window_size, self.window_size + 1):
                    context_word_pos = center_word_pos + w
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    assert len(context_word_idx) == 1
                    context_word_idx = context_word_idx[0][0]
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

    corpus, keywords = get_raw_data()
    assert len(corpus) == len(keywords)

    ######### for debugging #########
    max_len = 1
    assert max_len <= len(corpus)
    corpus = corpus[:max_len]
    keywords = keywords[:max_len]
    #################################

    data_loader = DataLoader()
    data_loader.load_data(corpus, keywords)
    X, y = data_loader.get_data()    
    vocab, inv_vocab, n_vocab = data_loader.get_metadata()
    embedder = EmbeddingLayer(n_vocab, embedding_dim, window_size)
    embedder.gen_embeddings(dataset=X, epochs=epochs, lr=lr)
    X_emb = embedder.get_embeddings(X)
    y_emb = embedder.get_embeddings(X)

if __name__ == '__main__':
    main()
