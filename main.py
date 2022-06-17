from model import *
from load_data import *
from torch.utils.data import DataLoader


BASE_DIR = './MIDI-BERT-CP/Data/CP_data/'

train_set = Dataset(BASE_DIR, 'train')
valid_set = Dataset(BASE_DIR, 'valid')
test_set = Dataset(BASE_DIR, 'test')

vocab = create_vocab(train_set, valid_set, test_set)

# batch_size=1 (8에서 update)
train_loader = DataLoader(train_set, 1, True)
valid_loader = DataLoader(valid_set, 1, False)
test_loader = DataLoader(test_set, 1, False)

mg_model = MG_model(vocab)

x, y = next(iter(train_loader))

embedding = mg_model(x)

#이거 실행시키면 0 기준으로 잘림,,,
#cutting(embedding)

