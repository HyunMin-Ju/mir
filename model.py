import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphMultisetTransformer, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Normal
import numpy as np
import math


#  piano tree vae -> https://arxiv.org/pdf/2008.07118.pdf
#  hierarchical message passing gnn -> https://arxiv.org/pdf/2009.03717.pdf

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
    def convert(self, x, beat_info):
        # beat = torch.select(x, dim=-1, index=1)
        # print(beat)
        # beat, ri, c = torch.unique_consecutive(beat, True, True)
        # print(beat)
        # print(ri)
        # print(c)
        # print(beat.shape)
        pitch = torch.select(x, dim=-1, index=2)
        dur = torch.select(x, dim=-1, index=3)

        '''
        print(x[0])
        print(pitch[0])
        print(beat[0])
        print(dur[0])
        '''

        beat = beat_info.cpu().apply_(lambda val: self.vocab['beat2idx'].get(val, 0)).cuda()
        pitch = pitch.cpu().apply_(lambda val: self.vocab['pitch2idx'].get(val, 0)).cuda()
        dur = dur.cpu().apply_(lambda val: self.vocab['dur2idx'].get(val, 0)).cuda()

        return pitch, beat, dur

    def forward(self, x, beat_info):
        p, b, d = self.convert(x, beat_info)

        pe = self.pitch_embedding(p)
        be = self.beat_embedding(b)
        de = self.dur_embedding(d)

        return torch.cat((pe, de), dim=1), be, [b, p, d]


class MG_model(nn.Module):
    def __init__(self, vocab,
                 emb_size=128, sim_hid_size=256, bar_hid_size=512, z_size=512, device=None):
        super(MG_model, self).__init__()
        self.vocab = vocab
        self.emb_layer = CPEmbeddingLayer(vocab, emb_size)

        self.sim_note_encoder = SAGEConv(emb_size * 2, sim_hid_size, aggr='mean', normalize=True)

        # Encoder Modules
        self.bar_encoder = nn.GRU(emb_size + sim_hid_size, bar_hid_size)
        self.bar_attention = nn.MultiheadAttention(bar_hid_size, num_heads=2, batch_first=True, dropout=0.4)

        self.bar_gnn = GCNConv(bar_hid_size, bar_hid_size)
        self.bar_readout = GraphMultisetTransformer(bar_hid_size, bar_hid_size, z_size)

        self.linear_mu = nn.Linear(z_size, z_size)
        self.linear_std = nn.Linear(z_size, z_size)

        # Decoder Modules

        self.dec_z_in_size = 256
        self.dec_z_hid_size = 1024
        self.dec_sim_hid_size = sim_hid_size + emb_size
        self.dec_note_hid_size = 512

        # self.dec_init_input = nn.Parameter(torch.rand(self.dec_bar_hid_size))
        self.sim_sos = nn.Parameter(torch.rand(self.dec_sim_hid_size))
        self.note_sos = nn.Parameter(torch.rand(emb_size*2))

        self.z2dec_hid = nn.Linear(z_size, self.dec_z_hid_size)
        self.z2dec_in = nn.Linear(z_size, self.dec_z_in_size)
        # self.bar_decoder = nn.GRU(self.dec_bar_hid_size + self.dec_z_in_size, self.dec_bar_hid_size, batch_first=True)

        self.sim_note_decoder = nn.GRU(self.dec_z_in_size + self.dec_sim_hid_size, self.dec_z_hid_size,
                                       batch_first=True)
        # self.sim_note_hid_linear = nn.Linear(self.dec_bar_hid_size, self.dec_sim_hid_size)
        self.beat_linear = nn.Linear(self.dec_z_hid_size, self.emb_layer.beat_num)
        self.note_decoder = nn.GRU(self.dec_z_hid_size + emb_size*2, self.dec_note_hid_size,
                                   batch_first=True)
        # self.note_attention = nn.Parameter(torch.zeros(3), requires_grad=True)
        # self.bar2note = nn.Linear(self.dec_bar_hid_size, self.dec_note_hid_size)
        # self.sim2note = nn.Linear(self.dec_sim_hid_size, self.dec_note_hid_size)
        self.note_hid_linear = nn.Linear(self.dec_z_hid_size, self.dec_note_hid_size)
        self.pitch_linear = nn.Linear(self.dec_note_hid_size, self.emb_layer.pitch_num)
        self.dur_linear = nn.Linear(self.dec_note_hid_size, self.emb_layer.dur_num)
        # self.bar2sim_note_gen = nn.GRU()

        # self.dec_init_input = nn.Parameter(torch.rand(2 * self.dec_emb_hid_size))

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # loss
        self.pitch_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.dur_loss = nn.CrossEntropyLoss()


    # 0: bar start, 1: bar continue
    def bar_cutting(self, x):
        bar_info = x[:, 0]
        mask = (bar_info == 0).nonzero().squeeze(0)
        # print(mask)
        indices = mask.cpu().numpy()
        mask2 = (bar_info == 2).nonzero().squeeze(0)
        if mask2.size(0) == 0:
            indices = np.append(indices, bar_info.size(0))
            x_wo_pad = x[:]
        else:
            indices = np.append(indices, mask2[0].item())
            x_wo_pad = x[:indices[-1]]
        if indices[0] != 0:
            indices = np.insert(indices, 0, 0)

        # print("indices: ", indices)
        sim_indices = []
        bar_indices = []
        beat_info = []
        adjacent_matrix = torch.zeros((indices[-1], indices[-1]), dtype=torch.long)
        for i in range(len(indices) - 1):
            s, e = indices[i], indices[i + 1]
            edge_info = x_wo_pad[s:e, 1]
            unique = torch.unique_consecutive(edge_info)
            beat_info.append(unique)
            bar_indices.append(unique.size(0))
            for j in range(len(unique)):
                u = torch.select(unique, 0, j)
                d = torch.where(edge_info == u, 1, 0).nonzero().view(-1)
                sim_s, sim_e = s + d[0], s + d[-1] + 1
                adjacent_matrix[sim_s:sim_e, sim_s:sim_e] = 1
                sim_indices.append((sim_s, sim_e))
        adjacent_matrix, _ = dense_to_sparse(adjacent_matrix)
        # bar_indices.insert(0, 0)
        bar_indices = torch.LongTensor(bar_indices)
        bar_indices = torch.cumsum(bar_indices, dim=0) - 1
        beat_info = torch.hstack(beat_info)
        return adjacent_matrix.cuda(), x_wo_pad, bar_indices.cuda(), sim_indices, beat_info

    def forward(self, x):

        # Encoder
        adj_mat, x_wo_pad, bar_indices, sim_indices, beat_info = self.bar_cutting(x)
        # single note embedding
        pd_emb, beat_emb, gt = self.emb_layer(x_wo_pad, beat_info)
        # sim note embedding
        # print(pd_emb.shape)
        out = self.sim_note_encoder(pd_emb, adj_mat)
        out = F.leaky_relu(out, 0.2)
        sim_note_emb = []
        for idx in sim_indices:
            s_s, s_e = idx
            sim_emb = torch.mean(out[s_s:s_e], dim=0)
            sim_note_emb.append(sim_emb)
        sim_note_emb = torch.stack(sim_note_emb)
        sim_note_emb = torch.hstack((sim_note_emb, beat_emb))

        out, h_n = self.bar_encoder(sim_note_emb.unsqueeze(1))
        bar_emb = torch.index_select(out.squeeze(1), 0, bar_indices).unsqueeze(0)

        _, weights = self.bar_attention(bar_emb, bar_emb, bar_emb)
        bar_num = weights.size(1)
        weights = weights.squeeze(0).view(-1)
        k = math.floor(weights.size(0) * 0.5)
        values, indices = torch.topk(weights, k)
        #
        edge_indices = torch.stack([torch.div(indices, bar_num, rounding_mode='floor'),
                                    torch.remainder(indices, bar_num)]).long().cuda()
        bgg = self.bar_gnn(bar_emb.squeeze(0), edge_indices, values)
        bgg = F.leaky_relu(bgg, 0.2)
        # print(bgg.shape)
        # score_emb = self.bar_readout(bgg, torch.zeros(bgg.size(0), dtype=torch.int64).cuda(), edge_indices)
        score_emb = F.leaky_relu(global_mean_pool(bgg, torch.zeros(bgg.size(0), dtype=torch.int64).cuda()), 0.2)
        mu = self.linear_mu(score_emb)
        std = self.linear_std(score_emb).exp_()
        # print(mu, std)
        dist = Normal(mu, std)

        dist_ = dist.rsample()

        # Decoder
        z_hid = self.z2dec_hid(dist_).unsqueeze(0)
        z_in = self.z2dec_in(dist_).unsqueeze(0)

        # token = self.dec_init_input.unsqueeze(0).unsqueeze(0)

        beats_list = []
        pitchs_list = []
        durs_list = []
        bar_xlike = []
        beat_xlike = []
        prev_b_idx = 0
        for t in range(bar_num):
            # bar_emb_dec, z_hid = self.bar_decoder(torch.cat([token, z_in], dim=-1), z_hid)
            b_idx = bar_indices[t]+1
            beats, pitchs, durs, _beat_xlike = self.decode_sim((z_in, z_hid), sim_note_emb[prev_b_idx:b_idx], sim_indices[prev_b_idx:b_idx], pd_emb, gt)
            beats_list.append(beats)
            pitchs_list.append(pitchs)
            durs_list.append(durs)
            # token = bar_emb[0, t].view(1, 1, -1)
            # token = bar_emb_dec
            prev_b_idx = b_idx

            bar_info = torch.ones_like(_beat_xlike)
            bar_info[0] = 0

            beat_xlike.append(_beat_xlike)
            bar_xlike.append(bar_info)

        pred_beat = torch.cat(beats_list, 0).squeeze(1)
        pred_pitch = torch.cat(pitchs_list, 0)
        pred_dur = torch.cat(durs_list)
        # print(pred_beat.shape)
        # print(pred_pitch.shape)
        # print(pred_dur.shape)
        bar_xlike = torch.cat(bar_xlike, 0)
        beat_xlike = torch.cat(beat_xlike, 0)
        pitch_xlike = torch.argmax(pred_pitch, dim=1).cpu()
        # pitch_xlike = pitch_xlike.apply_(lambda val: self.vocab['idx2pitch'][val])
        dur_xlike = torch.argmax(pred_dur, dim=1).cpu()
        # dur_xlike = dur_xlike.apply_(lambda val: self.vocab['idx2dur'][val])
        pred_xlike = torch.stack([bar_xlike, beat_xlike, pitch_xlike, dur_xlike]).long().numpy()
        # pred_xlike = None

        beat_loss = self.dur_loss(pred_beat, gt[0])
        pitch_loss = self.pitch_loss(pred_pitch, gt[1])
        dur_loss = self.dur_loss(pred_dur, gt[2])
        # print(beat_loss, pitch_loss, dur_loss)
        return beat_loss, pitch_loss, dur_loss, dist, pred_xlike

    def decode_sim(self, bar_emb_dec, sim_note_emb, sim_indices, orig_pd, gt):

        token = self.sim_sos.view(1, 1, -1)
        z_in, z_hid = bar_emb_dec
        # print(bar_emb_dec.shape)
        # sim_note_hid = self.sim_note_hid_linear(bar_emb_dec)
        # bar_info = self.bar2note(bar_emb_dec)
        bar_info = None
        beat_list = []
        pitch_list = []
        dur_list = []
        beat_xlike = []
        # print(sim_note_emb.shape)
        for i, sim_note in enumerate(sim_note_emb):
            sim_emb_dec, z_hid = self.sim_note_decoder(torch.cat([token, z_in], dim=-1), z_hid)

            s, e = sim_indices[i]
            _pitch_list, _dur_list = self.decode_note(bar_info, sim_emb_dec, orig_pd[s:e])
            pitch_list += _pitch_list
            dur_list += _dur_list

            token = sim_note.view(1, 1, -1)
            # token = sim_emb_dec
            beat = self.beat_linear(sim_emb_dec)
            beat_list.append(beat)

            beat_info = torch.ones(len(_pitch_list))
            real_beat = self.vocab['idx2beat'][torch.argmax(beat).cpu().detach().item()]
            beat_info *= real_beat
            beat_xlike.append(beat_info)


        return torch.cat(beat_list, 0), torch.cat(pitch_list, 0), torch.cat(dur_list, 0), torch.cat(beat_xlike, 0)

    def decode_note(self, bar_info, sim_emb_dec, note_gt):

        token = self.note_sos.view(1, 1, -1)
        note_hid = self.note_hid_linear(sim_emb_dec)

        pitch_list, dur_list = [], []
        # sim_info = self.sim2note(sim_emb_dec)

        for note in note_gt:
            note_emb, note_hid = self.note_decoder(torch.cat([token, sim_emb_dec], dim=-1),
                                                   note_hid)
            token = note.view(1, 1, -1)

            # att_weight = torch.softmax(self.note_attention, dim=-1)
            # final_emb = torch.stack([bar_info, sim_info, note_emb]).view(3, -1)
            # final_emb = torch.matmul(att_weight, final_emb)
            note_emb = note_emb.squeeze(0)

            pitch = self.pitch_linear(note_emb)
            dur = self.dur_linear(note_emb)
            pitch_list.append(pitch)
            dur_list.append(dur)
            # print(pitch.shape, dur.shape)
            # print(pitch, dur)
        return pitch_list, dur_list
