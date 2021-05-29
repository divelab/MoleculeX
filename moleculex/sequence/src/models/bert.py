import torch.nn as nn
from .bert_embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, activation='gelu', seq_len=512):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        # self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, seq_len=seq_len)

        # # multi-layers transformer blocks, deep network
        # self.transformer_blocks = nn.ModuleList(
        #     [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        # Using Pytorch TransformerEncoder
        self.encoders = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=attn_heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation=activation,
            ),
            num_layers=n_layers,
        )

    # def forward(self, x, segment_info):
    def forward(self, x, segment_label=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask = (x == 0)  # I think here is different from our old version... Please help to confirm~~~

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_label)

        # # running over multiple transformer blocks
        # for transformer in self.transformer_blocks:
        #     x = transformer.forward(x, mask)

        # Using Pytorch TransformerEncoder
        x = self.encoders(x.permute(1, 0, 2), src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)

        return x
