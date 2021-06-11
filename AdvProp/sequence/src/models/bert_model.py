import torch
import torch.nn as nn
from .bert import BERT

def save_bert_model(bert_model, model_pth):
    dict = {}
    embed = {}
    embed['token'] = bert_model.embedding.token.state_dict()
    if hasattr(bert_model.embedding, 'segment'):
        embed['segment'] = bert_model.embedding.segment.state_dict()
    embed['norm'] = bert_model.embedding.norm.state_dict()
    dict['embedding'] = embed
    dict['encoders'] = bert_model.encoders.state_dict()
    torch.save(dict, model_pth)


def load_bert_model(bert_model, model_pth):
    dict = torch.load(model_pth)
    embed = dict['embedding']
    bert_model.embedding.token.load_state_dict(embed['token'])
    if hasattr(bert_model.embedding, 'segment'):
        bert_model.embedding.segment.load_state_dict(embed['segment'])
    bert_model.embedding.norm.load_state_dict(embed['norm'])
    bert_model.encoders.load_state_dict(dict['encoders'])


class BERTChem_TAR(nn.Module):
    def __init__(self, task='cls', n_out=1, vocab_size=70, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, activation='gelu', seq_len=512):
        super().__init__()
        self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, dropout, activation, seq_len)
        if task == 'cls':
            self.pred = nn.Sequential(
                Chem_TAR(self.bert.hidden, n_out),
                nn.Sigmoid()
            )
        elif task == 'reg':
            self.pred = nn.Sequential(
                Chem_TAR(self.bert.hidden, n_out)
            )

    def forward(self, x):
        return self.pred(self.bert(x))

    def load_feat_net(self, load_pth):
        load_bert_model(self.bert, load_pth)

    def save_model(self, save_pth):
        dict = {}
        embed = {}
        embed['token'] = self.bert.embedding.token.state_dict()
        if hasattr(self.bert.embedding, 'segment'):
            embed['segment'] = self.bert.embedding.segment.state_dict()
        embed['norm'] = self.bert.embedding.norm.state_dict()
        dict['embedding'] = embed
        dict['encoders'] = self.bert.encoders.state_dict()
        dict['pred'] = self.pred.state_dict()
        torch.save(dict, save_pth)

    def load_model(self, load_pth):
        dict = torch.load(load_pth)
        embed = dict['embedding']
        self.bert.embedding.token.load_state_dict(embed['token'])
        if hasattr(self.bert.embedding, 'segment'):
            self.bert.embedding.segment.load_state_dict(embed['segment'])
        self.bert.embedding.norm.load_state_dict(embed['norm'])
        self.bert.encoders.load_state_dict(dict['encoders'])
        self.pred.load_state_dict(dict['pred'])


class Chem_TAR(nn.Module):
    def __init__(self, hidden, n_out):
        super().__init__()
        self.linear = nn.Linear(hidden, n_out, bias=True)

    def forward(self, x):
        feature = x[:,0]
        return self.linear(feature)


class BERTChem_Mask(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, activation='gelu', seq_len=512):
        super().__init__()
        self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, dropout, activation, seq_len)
        self.mask_chem = MaskedChem(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        return self.mask_chem(x)

    def save_feat_net(self, save_pth):
        save_bert_model(self.bert, save_pth)

    def load_feat_net(self, load_pth):
        load_bert_model(self.bert, load_pth)


class MaskedChem(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class Con_atom(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, activation='gelu', seq_len=512):
        super(Con_atom, self).__init__()
        self.encoder = BERT(vocab_size, hidden, n_layers, attn_heads, dropout, activation, seq_len)

    def save_feat_net(self, save_pth):
        save_bert_model(self.encoder, save_pth)

    def forward(self, seq, seq_mask):
        out_mask = self.encoder(seq_mask)
        out = self.encoder.embedding(seq)
        return out, out_mask