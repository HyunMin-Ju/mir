import torch
import torch.nn as nn
import torch.nn.functional as F


class CPEmbeddingLayer(nn.Module):
    def __init__(self, vocab, emb_size=128):
        super(CPEmbeddingLayer, self).__init__()

        self.vocab = vocab
        self.pitch_num = len(vocab['idx2pitch'])
        self.beat_num = len(vocab['idx2beat'])
        self.dur_num = len(vocab['idx2dur'])

        self.pitch_embedding = nn.Embedding(self.pitch_num, emb_size)
        self.beat_embedding = nn.Embedding(self.beat_num, emb_size)
        self.dur_embedding = nn.Embedding(self.dur_num, emb_size)

    # _vocabs = {'idx2pitch', 'idx2sub', 'idx2dur', 'pitch2idx', 'sub2idx', 'dur2idx'}
    def convert(self, x):
        pitch = torch.select(x, dim=-1, index=1)
        beat = torch.select(x, dim=-1, index=2)
        dur = torch.select(x, dim=-1, index=3)

        '''
        print(x[0])
        print(pitch[0])
        print(beat[0])
        print(dur[0])
        '''

        pitch = pitch.cpu().apply_(lambda val: self.vocab['pitch2idx'].get(val, 0))
        beat = beat.cpu().apply_(lambda val: self.vocab['beat2idx'].get(val, 0))
        dur = dur.cpu().apply_(lambda val: self.vocab['dur2idx'].get(val, 0))

        return pitch, beat, dur

    def forward(self, x):

        p, b, d = self.convert(x)

        pe = self.pitch_embedding(p)
        se = self.beat_embedding(b)
        de = self.dur_embedding(d)

        note_embedding = torch.cat((pe, se, de), dim=-1)

        return note_embedding

class MG_model(nn.Module):
    def __init__(self, vocab):
        super(MG_model, self).__init__()

        self.emb_layer = CPEmbeddingLayer(vocab)

    def forward(self, x):

        emb = self.emb_layer(x)
        #print(emb.shape)

        #print('bar information: ', x[:,:,0][0])
        return x[:,:,0][0]

# 0: bar start, 1: bar continue
def cutting(x):
    result = []
    i = 0
    while(i < len(x)):
        if x[i]==0:
            tmp = [0]
            j = i+1
            while(j < len(x)):
                #여기서 tensor(1)이라고 나와서 바꿈
                if x[j] == 1 :
                    tmp.append(1)
                    j += 1
                #이건 tensor(2),,,
                elif x[j] == 2:
                    tmp.append(2)
                    j += 1
                else:
                    break
            result.append(tmp)
            i += 1
        else:
            i += 1
            continue
    print(result)
    return 0