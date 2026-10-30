"""Predictor (world model), instruction-conditioned scorer, and direct classifier baseline."""
import torch
import torch.nn as nn

FEAT_DIM, N_TOK, TEXT_DIM, ACT_DIM, MAX_PUSH = 384, 256, 384, 4, 2
ACT_SCALE = 5.0  # push coords are in meters, roughly +-0.3


def _encoder(d, layers, heads):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def _mlp(i, d, o):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class Predictor(nn.Module):
    """z_t (B,256,384) + push a_t (B,4) -> z_{t+1}. Residual prediction over patch tokens."""

    def __init__(self, d=256, layers=6, heads=8):
        super().__init__()
        self.inp = nn.Linear(FEAT_DIM, d)
        self.pos = nn.Parameter(torch.randn(1, N_TOK, d) * 0.02)
        self.act = _mlp(ACT_DIM, d, d)
        self.body = _encoder(d, layers, heads)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, FEAT_DIM))
        nn.init.zeros_(self.out[1].weight), nn.init.zeros_(self.out[1].bias)

    def forward(self, z, a):
        a = self.act(a * ACT_SCALE).unsqueeze(1)
        h = self.inp(z) + self.pos + a  # action added to every patch token
        h = self.body(torch.cat([a, h], 1))[:, 1:]  # plus a separate action token
        return z + self.out(h)

    def rollout(self, z, seq, n_push):
        """Apply pushes in seq (B,MAX_PUSH,4) autoregressively; stop per sample at n_push."""
        for k in range(seq.shape[1]):
            active = (n_push > k).view(-1, 1, 1)
            if not active.any():
                break
            z = torch.where(active, self(z, seq[:, k]), z)
        return z


class Scorer(nn.Module):
    """Terminal patch tokens + instruction embedding -> success logit."""

    def __init__(self, d=256, layers=2, heads=8):
        super().__init__()
        self.inp = nn.Linear(FEAT_DIM, d)
        self.pos = nn.Parameter(torch.randn(1, N_TOK, d) * 0.02)
        self.txt = _mlp(TEXT_DIM, d, d)
        self.body = _encoder(d, layers, heads)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, z, text):
        t = self.txt(text).unsqueeze(1)
        h = self.body(torch.cat([t, self.inp(z) + self.pos], 1))
        return self.head(h[:, 0]).squeeze(-1)


class DirectClassifier(nn.Module):
    """Start patch tokens + action sequence + instruction -> success logit. No future prediction."""

    def __init__(self, d=256, layers=6, heads=8):
        super().__init__()
        self.inp = nn.Linear(FEAT_DIM, d)
        self.pos = nn.Parameter(torch.randn(1, N_TOK, d) * 0.02)
        self.act = _mlp(ACT_DIM, d, d)
        self.step = nn.Parameter(torch.randn(1, MAX_PUSH, d) * 0.02)
        self.txt = _mlp(TEXT_DIM, d, d)
        self.body = _encoder(d, layers, heads)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, z, seq, n_push, text):
        a = self.act(seq * ACT_SCALE) + self.step  # (B,MAX_PUSH,d)
        t = self.txt(text).unsqueeze(1)
        h = torch.cat([t, a, self.inp(z) + self.pos], 1)
        pad = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
        pad[:, 1:1 + MAX_PUSH] = torch.arange(MAX_PUSH, device=h.device) >= n_push.unsqueeze(1)
        h = self.body(h, src_key_padding_mask=pad)
        return self.head(h[:, 0]).squeeze(-1)
