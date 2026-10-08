import abc
import torch
from typing import Optional

class singleSelfAttention(torch.nn.Module):
    """
    Simple single head self-attention class
    """

    def __init__(self, n_in: int, n_out: int, drop_out: float = 0., bias: bool = False):
        super(singleSelfAttention, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.Wq = torch.nn.Linear(n_in, n_out, bias=bias)
        self.Wk = torch.nn.Linear(n_in, n_out, bias=bias)
        self.Wv = torch.nn.Linear(n_in, n_out, bias=bias)

        if drop_out > 0.:
            self.dropout = torch.nn.Dropout(drop_out)
        else:
            self.dropout = None

    def forward(self, y):
        # y -> b X t X n_in
        que = self.Wq(y)  # y @ self.Wq  # b X t X n_out
        key = self.Wk(y)  # y @ self.Wk  # b X t X n_out
        val = self.Wv(y)  # y @ self.Wv  # b X t X n_out

        wts = que @ key.transpose(-2, -1)  # b X t X t
        att = torch.softmax(wts / self.n_out ** 0.5, dim=-1)  # softmax across keys
        if self.dropout:
            att = self.dropout(att)

        return att @ val  # b X t X n_out


class singleCausalAttention(torch.nn.Module):
    """
    Simple single head self-attention class with causal mask
    """

    def __init__(self, n_in: int, n_out: int, contextLength: int, drop_out: float = 0., bias: bool = False):
        super(singleCausalAttention, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.Wq = torch.nn.Linear(n_in, n_out, bias=bias)
        self.Wk = torch.nn.Linear(n_in, n_out, bias=bias)
        self.Wv = torch.nn.Linear(n_in, n_out, bias=bias)

        if drop_out > 0.:
            self.dropout = torch.nn.Dropout(drop_out)
        else:
            self.dropout = None

        self.contextLength = contextLength
        self.register_buffer('mask', torch.triu(torch.ones(contextLength, contextLength), diagonal=1).bool())

    def forward(self, y):
        # y -> b X t X n_in
        # check size isn't greater than context length
        nTokens = y.shape[1]
        assert (nTokens <= self.contextLength)

        que = self.Wq(y)  # y @ self.Wq  # b X t X n_out
        key = self.Wk(y)  # y @ self.Wk  # b X t X n_out
        val = self.Wv(y)  # y @ self.Wv  # b X t X n_out

        wts = que @ key.transpose(-2, -1)  # b X t X t

        # set masked values to -infinity so softmax sets to 0 for causality
        wts.masked_fill_(self.mask[:nTokens, :nTokens], -torch.inf)

        att = torch.softmax(wts / self.n_out ** 0.5, dim=-1)  # softmax across keys
        if self.dropout:
            att = self.dropout(att)

        return att @ val  # b X t X n_out


class multiHeadAttention(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, nHeads: int, contextLength: Optional[int] = None, drop_out: float = 0.,
                 bias: bool = False, is_causal: bool = True):
        super(multiHeadAttention, self).__init__()

        # output dimension must be a multiple of n_heads
        assert (n_out % nHeads == 0)
        assert (not is_causal) or (contextLength is not None), "contextLength required for causal attention"

        self.n_in = n_in
        self.n_out = n_out
        self.nHeads = nHeads
        self.dHead = n_out // nHeads
        self.contextLength = contextLength
        self.is_causal = is_causal

        self.Wqkv = torch.nn.Linear(n_in, 3 * n_out, bias=bias)
        self.ff = torch.nn.Linear(n_out, n_out)  # linear layer to combine outputs

        if drop_out > 0.:
            self.dropout = torch.nn.Dropout(drop_out)
        else:
            self.dropout = None

        if is_causal:
            self.register_buffer("mask", torch.triu(torch.ones(contextLength, contextLength), diagonal=1).bool())

    def forward(self, y):
        b, nTokens, _ = y.shape

        qkv = self.Wqkv(y)  # b X nTokens X 3*n_out
        qkv = qkv.view(b, nTokens, 3, self.nHeads, self.dHead)  # b x nTokens x 3 x nHeads x dHead
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x b x nHeads x nTokens x dHead
        que, key, val = qkv  # each is b x nHeads x nTokens x dHead

        # Compute scaled dot-product attention (aka self-attention) with causal masking
        wts = que @ key.transpose(-2, -1)  # b x nHeads x nTokens x nTokens

        # Use the mask to fill attention scores
        if self.is_causal:
            wts.masked_fill_(self.mask[:nTokens, :nTokens], -torch.inf)

        att = torch.softmax(wts / self.dHead ** 0.5, dim=-1)  # softmax across keys
        if self.dropout:
            att = self.dropout(att)

        context = (att @ val).transpose(1, 2)  # b X nTokens X nHeads X dHead

        # Combine heads as n_out = nHeads * dHead
        context = context.contiguous().view(b, nTokens, self.n_out)
        return self.ff(context)  # optional projection


class multiHeadAttentionTorch(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, nHeads: int, contextLength: Optional[int] = None, drop_out: float = 0.,
                 bias: bool = False, is_causal: bool = True, needWts: bool = True):
        super(multiHeadAttentionTorch, self).__init__()

        # output dimension must be a multiple of n_heads
        assert (n_out % nHeads == 0)
        assert (not is_causal) or (contextLength is not None), "contextLength required for causal attention"

        self.n_in = n_in
        self.n_out = n_out
        self.nHeads = nHeads
        self.dHead = n_out // nHeads
        self.contextLength = contextLength
        self.is_causal = is_causal
        self.needWeights = needWts

        self.mha = torch.nn.MultiheadAttention(embed_dim=n_out, num_heads=nHeads, dropout=drop_out,
                                               bias=bias, add_bias_kv=bias, batch_first=True)

        self.ff = torch.nn.Linear(n_out, n_out)  # linear layer to combine outputs
        if is_causal:
            self.register_buffer("mask", torch.triu(torch.ones(contextLength, contextLength), diagonal=1).bool())

    def forward(self, y):
        b, nTokens, _ = y.shape

        if self.is_causal:
            assert self.contextLength >= nTokens
            mask = self.mask[:nTokens, :nTokens]
            context, _ = self.mha(y, y, y, attn_mask=mask, need_weights=self.needWeights)
        else:
            context, _ = self.mha(y, y, y, need_weights=self.needWeights)

        return self.ff(context)  # optional projection


class multiHeadAttentionTorchSDP(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, nHeads: int, contextLength: Optional[int] = None, dropout: float = 0., bias: bool = False, is_causal: bool = True):
        super(multiHeadAttentionTorchSDP, self).__init__()

        # output dimension must be a multiple of n_heads
        assert (n_out % nHeads == 0)
        assert (not is_causal) or (contextLength is not None), "contextLength required for causal attention"

        self.n_in = n_in
        self.n_out = n_out
        self.nHeads = nHeads
        self.dHead = n_out // nHeads
        self.contextLength = contextLength
        self.dropout = dropout
        self.is_causal = is_causal

        self.Wqkv = torch.nn.Linear(n_in, 3 * n_out, bias=bias)
        self.ff = torch.nn.Linear(n_out, n_out)  # linear layer to combine outputs

    def forward(self, y):
        b, nTokens, _ = y.shape

        qkv = self.Wqkv(y)  # b X nTokens X 3*nout
        qkv = qkv.view(b, nTokens, 3, self.nHeads, self.dHead)  # b x nTokens x 3 x nHeads x dHead
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x b x nHeads x nTokens x dHead
        que, key, val = qkv  # each is b x nHeads x nTokens x dHead

        dropout = 0. if not self.training else self.dropout

        context = torch.nn.functional.scaled_dot_product_attention(que, key, val,
                                                                   dropout_p=dropout,
                                                                   is_causal=self.is_causal)  # b x nHeads x nTokens x dHead

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context = context.transpose(1, 2).contiguous().view(b, nTokens, self.n_out)
        return self.ff(context)  # optional projection


if __name__ == "__main__":
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running on {device}")

    batch_size = 8
    context_len = 1024
    embed_dim = 768
    n_heads = 12
    dropout = 0.0
    bias = False

    embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

    mha = multiHeadAttention(embed_dim, embed_dim, n_heads,
                             context_len, dropout, bias, True).to(device)
    out = mha(embeddings)
    print(out.shape)

    mha = multiHeadAttention(embed_dim, embed_dim, n_heads,
                             context_len, dropout, bias, False).to(device)
    out = mha(embeddings)
    print(out.shape)

    mha = multiHeadAttentionTorch(embed_dim, embed_dim, n_heads,
                                  context_len, dropout, bias, False).to(device)
    out = mha(embeddings)
    print(out.shape)

    mha = multiHeadAttentionTorch(embed_dim, embed_dim, n_heads,
                                  context_len, dropout, bias, True).to(device)
    out = mha(embeddings)
    print(out.shape)

    mha = multiHeadAttentionTorchSDP(embed_dim, embed_dim, n_heads,
                                     context_len, dropout, bias).to(device)
    out = mha(embeddings)
    print(out.shape)
