import torch
import math
import torch.nn as nn


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1, seq_len=512):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=seq_len)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    # def forward(self, sequence, segment_label):
    def forward(self, sequence, segment_label=None):
        x = self.token(sequence)
        x += self.position(sequence)
        return self.dropout(self.norm(x))


class TokenEmbedding(nn.Embedding):
    # def __init__(self, vocab_size, embed_size=512):
    def __init__(self, vocab_size, embed_size=200):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        return self.pe[:, :x.size(1)]