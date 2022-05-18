import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

'''
DATA_PATH_train = 'MIDI-BERT-CP/Data/CP_data/pop909_train.npy'
train_data = np.load(DATA_PATH_train)

DATA_PATH_test = 'MIDI-BERT-CP/Data/CP_data/pop909_test.npy'
test_data = np.load(DATA_PATH_test)

DATA_PATH_valid = 'MIDI-BERT-CP/Data/CP_data/pop909_valid.npy'
valid_data = np.load(DATA_PATH_valid)

#dataloader로 만들어서 getitem
init에서 path가 주어지면 해당 데이터 불러오기

init안에서 path로 파일을 불러오고
self.x = <-파일로드
y=ans
keyword='train

getitem은 self.x의 idx

토치/텐서로 바꿔서
'''

class Dataset:
    def __init__(self,bas_dir, keyword='train'):
        x_path = bas_dir+'pop909_'+keyword+'.npy'
        y_path = bas_dir+  'pop909_'+keyword+'_melans.npy'

        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloader(base_dir, keyword, batch_size, shuffle):
    dataset = Dataset(base_dir,keyword)
    loader =DataLoader(dataset,batch_size, shuffle)
    return loader

BASE_DIR = 'MIDI-BERT-CP/Data/CP_data/'

class Dataset_2:
    def __init__(self, muspy_dataset, vocabs=None):
        self.dataset =
        if vocabs is None:
            self.idx2pitch, self.idx2sub, self.idx2dur = self._get_vocab_info()
            self.idx2pitch += ['start', 'end']
            self.idx2sub += ['start', 'end']
            self.idx2dur += ['start', 'end']
            self.pitch2idx = {x: i for i, x in enumerate(self.idx2pitch)}
            self.sub2idx = {x: i for i, x in enumerate(self.idx2sub)}
            self.dur2idx = {x: i for i, x in enumerate(self.idx2dur)}
        else:
            self.idx2pitch, self.idx2sub, self.idx2dur, self.pitch2idx, self.sub2idx, self.dur2idx = vocabs

    def _get_vocab_info(self):
        entire_pitch = []
        entire_sub = []
        entire_dur = []
        for note_rep in self.dataset:
            pitch_in_piece = note_rep[:, 1]
            sub_in_piece = note_rep[:, 2]
            dur_in_piece = note_rep[:, 3]
            entire_pitch += pitch_in_piece.tolist()
            entire_sub += sub_in_piece.tolist()
            entire_dur += dur_in_piece.tolist()
        return list(set(entire_pitch)), list(set(entire_sub)), list(set(entire_dur))

#batch_size=8
train_load = get_dataloader(BASE_DIR,'train',8,True)
valid_load = get_dataloader(BASE_DIR, 'valid', 8, True)
test_load = get_dataloader(BASE_DIR, 'test', 8, True)
#print(train_load)

batch = next(iter(train_load))
#print(batch)
train_set = Dataset(BASE_DIR)
dataset2 = Dataset_2(train_set)
#nn.Embedding(num_embeddings, embedding_dim)
# pitch_embedding = nn.Embedding( ,4)
# subbeat_embedding = nn.Embedding( , 4)
# dur_embedding = nn.Embedding( , 4)

#torch.cat(tensors, dim)
# note_embedding = torch.cat((pitch_embedding, subbeat_embedding, dur_embedding), dim=0)

#print(note_embedding)
#print(train_embedding)

