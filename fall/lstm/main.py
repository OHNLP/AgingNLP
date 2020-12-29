import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import training as t
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb

parser = argparse.ArgumentParser(description='Bi-LSTM text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
# data 
parser.add_argument('-train-test-ratio', type=int, default=0.10, help='ratio of test data for testing [default: 0.10]')
parser.add_argument('-train-vali-ratio', type=int, default=0.80, help='ratio of validation data for training [default: 0.80]')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

raw_data_path = '/data.tsv'
destination_folder = '/content'

train_test_ratio = args.train_test_ratio
train_valid_ratio = args.train_vali_ratio
first_n_words = 200

def trim_string(x):
    x = x.split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])
    return x


# Fields
def fall_data(docid_field, text_field, label_field, **kargs):
    fields = [('docid', docid_field), ('text', text_field), ('label', label_field)]
    train_data, valid_data, test_data = data.TabularDataset.splits(
											path = '/Fall/data/',
											train = 'training.tsv',
											validation = 'vali.tsv',
											test = 'test.tsv',											
											format = 'tsv',
											fields = fields,
											skip_header = True)

    word_embedding = "glove.6B.300d"
    text_field.build_vocab(train_data, valid_data, test_data, vectors = word_embedding)
    label_field.build_vocab(train_data, valid_data, test_data)
    
    print (len(text_field.vocab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1

	#create batch size iteration
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
			(train_data, valid_data, test_data),
			sort = False, #don't sort test/validation data
			batch_size=BATCH_SIZE,
			device=device)
    return train_iterator, valid_iterator, test_iterator, text_field.vocab

# load data
print("\nLoading data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_field = data.Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
label_field = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
docid_field = data.RawField()
train_iter, dev_iter, test_iter, text_voca = fall_data(docid_field, text_field, label_field, device=-1, repeat=False)

# train or eval
if args.test:
	best_model = model.LSTM().to(device)
	optimizer = optim.Adam(best_model.parameters(), lr=args.lr)
	t.load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
	t.evaluate(best_model, test_iter)
else:
    print('start training')
	wandb.init()
	wandb.watch(model)     
	model = model.LSTM(text_voca).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	eval_every = len(train_iter) // 2
	t.train(model=model, optimizer=optimizer, train_loader=train_iter, valid_loader=dev_iter, num_epochs=args.epochs, eval_every = eval_every, file_path= destination_folder, device=device)






