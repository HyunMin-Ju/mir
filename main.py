from model import *
from load_data import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.distributions import kl_divergence
from utils import *

BASE_DIR = '../MIDI-BERT-CP/Data/CP_data/'

train_set = Dataset(BASE_DIR, 'train')

vocab = create_vocab(train_set)

# batch_size=1 (8에서 update)
train_loader = DataLoader(train_set, 1, False)
# valid_loader = DataLoader(valid_set, 1, False)
# test_loader = DataLoader(test_set, 1, False)

mg_model = MG_model(vocab)
# for param in mg_model.parameters():
#     nn.init.xavier_uniform_(param)
mg_model.load_state_dict(torch.load('mg_model2_e4_s600.pt'))
mg_model.cuda()

lr = 1e-3
n_epoch = 40
clip = 0.8
optimizer = Adam(mg_model.parameters(), lr)
scheduler = MinExponentialLR(optimizer, gamma=0.99, minimum=1e-4)
normal = standard_normal(512)

f = open('train_log.txt', 'w')

for epoch in range(5, n_epoch+1):
    objs_b = AverageMeter()
    objs_p = AverageMeter()
    objs_d = AverageMeter()
    objs_k = AverageMeter()
    for i, batch in enumerate(train_loader):
        batch = batch.cuda()
        optimizer.zero_grad()
        b_loss, p_loss, d_loss, dist, pred_xlike = mg_model(batch.squeeze(0))
        kl = kl_divergence(dist, normal).mean()
        loss = 0.5 * b_loss + p_loss + d_loss + 0.1 * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mg_model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        objs_b.update(b_loss.item())
        objs_p.update(p_loss.item())
        objs_d.update(d_loss.item())
        objs_k.update(kl.item())

        print(f'EPOCH {epoch} / STEP {i} / beat loss: {b_loss} / pitch loss: {p_loss} / dur loss: {d_loss} / kl: {kl}')

        if i % 100 == 0:
            log = f'EPOCH {epoch} / STEP {i} / beat loss: {objs_b.avg} / pitch loss: {objs_p.avg} / dur loss: {objs_d.avg} / kl: {objs_k.avg}\n'
            print(log)
            f.write(log)

        if i % 300 == 0:
            torch.save(mg_model.state_dict(), f'mg_model2_e{epoch}_s{i}.pt')
f.close()

