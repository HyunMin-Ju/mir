import numpy as np
from torch.utils.data import DataLoader

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
keyword='trsin

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
train_load = get_dataloader(BASE_DIR,'train',8,True)
