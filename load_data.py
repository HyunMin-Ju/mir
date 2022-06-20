import numpy as np
import torch
import torch.nn as nn

'''
DATA_PATH_train = 'MIDI-BERT-CP/Data/CP_data/pop909_train.npy'
train_data = np.load(DATA_PATH_train)

DATA_PATH_test = 'MIDI-BERT-CP/Data/CP_data/pop909_test.npy'
test_data = np.load(DATA_PATH_test)

DATA_PATH_valid = 'MIDI-BERT-CP/Data/CP_data/pop909_valid.npy'
valid_data = np.load(DATA_PATH_valid)
'''


class Dataset:
    def __init__(self, bas_dir, keyword='train'):
        x_path = bas_dir + 'pop909_' + keyword + '.npy'
        y_path = bas_dir + 'pop909_' + keyword + '_melans.npy'

        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_vocab(train_set: Dataset, valid_set: Dataset, test_set: Dataset):
    whole_dataset = np.concatenate((train_set.x, valid_set.x, test_set.x), 0).reshape(-1, 4)

    entire_pitch = []
    entire_beat = []
    entire_dur = []

    pitch_in_piece = whole_dataset[:, 1]
    beat_in_piece = whole_dataset[:, 2]
    dur_in_piece = whole_dataset[:, 3]
    entire_pitch += pitch_in_piece.tolist()
    entire_beat += beat_in_piece.tolist()
    entire_dur += dur_in_piece.tolist()

    idx2pitch, idx2beat, idx2dur = list(set(entire_pitch)), list(set(entire_beat)), list(set(entire_dur))
    idx2pitch += ['start', 'end']
    idx2beat += ['start', 'end']
    idx2dur += ['start', 'end']
    pitch2idx = {x: i for i, x in enumerate(idx2pitch)}
    beat2idx = {x: i for i, x in enumerate(idx2beat)}
    dur2idx = {x: i for i, x in enumerate(idx2dur)}
    _vocabs = {'idx2pitch': idx2pitch, 'idx2beat': idx2beat, 'idx2dur': idx2dur, 'pitch2idx': pitch2idx,
               'beat2idx': beat2idx, 'dur2idx': dur2idx}
    return _vocabs

def note_represent(song):
    # whole_dataset = np.concatenate((train_set.x, valid_set.x, test_set.x), 0).reshape(-1, 4)
    x = song.squeeze(0).numpy().T
    print(x.shape)
    bar, sub_beat, pitch, dur = x[0], x[1], x[2], x[3]

    start_timestep = []

    for i in range(len(pitch)-1):
        if bar[i]==0:
            i += 1
        start_timestep.append(64 * i + (sub_beat[i] - 1) * 4)

    velocity = 64
    note_repr = (start_timestep, pitch, dur, velocity)

    return note_repr