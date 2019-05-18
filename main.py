from nltk.tokenize import word_tokenize
import numpy as np
import torch

class DataLoader:
    def __init__(self):
        pass        

    def load_data(self, verbose = True):
        toy_input = ['toy example', 'foo#', 'bar baz$ C++!', 'todo']
        toy_output = [['toy', 'easy'], ['foo'], ['C++', 'Java', 'Lists'], ['test']]
        
        tokenized_X = self._tokenize(dataset=toy_input)
        tokenized_y = self._tokenize(dataset=toy_output, keywords=True)
        self._construct_vocabulary(tokenized_X, tokenized_y)
        onehot_X = self._one_hot(dataset=tokenized_X)
        onehot_y = self._one_hot(dataset=tokenized_y, keywords=True)
        self.X = torch.tensor(onehot_X)
        self.y = torch.tensor(onehot_y)
        if verbose:
            print('====================tokenized====================')
            print(tokenized_X)
            print(tokenized_y)
            print('====================onehot====================')
            print(onehot_X)
            print(onehot_y)
            print('====================pytorch====================')
            print(self.X)
            print(self.y)

    def get_data(self):
        return self.X, self.y

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
        self.n_vocab = num_spec_tokens
        for sample in all_samples:
            for word in sample:
                if word not in self.vocab.keys():
                    self.vocab[word] = self.n_vocab
                    self.n_vocab += 1

def main():
    data_loader = DataLoader()
    data_loader.load_data()
    X, y = data_loader.get_data()    

if __name__ == '__main__':
    main()
