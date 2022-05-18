from model import *
from load_data import *


BASE_DIR = '../MIDI-BERT-CP/Data/CP_data/'

train_set = Dataset(BASE_DIR, 'train')
valid_set = Dataset(BASE_DIR, 'valid')
test_set = Dataset(BASE_DIR, 'test')

vocab = create_vocab(train_set, valid_set, test_set)

# batch_size=8
train_loader = DataLoader(train_set, 8, True)
valid_loader = DataLoader(valid_set, 8, False)
test_loader = DataLoader(test_set, 8, False)

emb_model = CPEmbeddingLayer(vocab)
x, y = next(iter(train_loader))

print(x.shape)

embedding = emb_model(x)
print(embedding.shape)