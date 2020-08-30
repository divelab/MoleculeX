import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU_ATT(nn.Module):
    def __init__(self, task='cls', n_out=1, vocab_size=70, embed_size=200, gru_size=200, att_size=200, num_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(input_size = embed_size, hidden_size=gru_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.att = nn.Sequential(
            nn.Linear(2*gru_size, att_size, bias=True),
            nn.Tanh(),
            nn.Linear(att_size, 1, bias=False),
            nn.Softmax(dim=1)
        )

        if task == 'reg':
            self.pred = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2*gru_size, n_out, bias=True)
            )
        elif task == 'cls':
            self.pred = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2*gru_size, n_out, bias=True),
                nn.Sigmoid()
            )
        elif task == 'con_seq':
            self.pred = nn.Sequential(
                nn.Linear(2*gru_size, 2*gru_size, bias=True),
                nn.ReLU(),
                nn.Linear(2*gru_size, n_out, bias=True)
            )
#         else:
#             raise ValueError('not supported task')

    def forward(self, x, x_lengths):
        total_length = x.shape[1]
        embed = self.embed(x)
        packed = pack_padded_sequence(embed, x_lengths, batch_first=True, enforce_sorted=False)
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(packed)
        padded_gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=total_length)
        att_score = self.att(padded_gru_out)
        feature = (padded_gru_out * att_score).sum(dim=1)

        if hasattr(self, 'pred'):
            out = self.pred(feature)
        else:
            out = feature
        return out

    def save_model(self, save_pth):
        torch.save(self.state_dict(), save_pth)

    def load_model(self, load_pth):
        self.load_state_dict(torch.load(load_pth))


if __name__ == '__main__':
    a = torch.tensor([[1,2,3,4,0]])
    model = GRU_ATT(vocab_size=10)
    out = model(a, [4])
    print(out.shape)