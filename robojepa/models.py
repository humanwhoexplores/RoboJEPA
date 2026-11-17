"""Predictor (world model), instruction-conditioned scorer, and direct classifier baseline."""
import torch
import torch.nn as nn

FEAT_DIM, N_TOK, TEXT_DIM, ACT_DIM, MAX_PUSH = 384, 256, 384, 4, 2
ACT_SCALE = 5.0  # push coords are in meters, roughly +-0.3


def _encoder(d, layers, heads, dropout=0.0):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
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


class TokenScorer(nn.Module):
    """Terminal patch tokens + instruction TOKEN sequence -> success logit, read from a CLS token.

    Unlike Scorer, the instruction is not pooled into one vector, so word order and words such as
    left/right survive and each word can attend to image patches."""

    def __init__(self, text_dim=TEXT_DIM, d=256, layers=4, heads=8, max_text=64, two_frame=False, feat_dim=FEAT_DIM):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d)  # feat_dim 1664: V-JEPA 2.1 ViT-G tokens (RoboJEPA)
        self.pos = nn.Parameter(torch.randn(1, N_TOK, d) * 0.02)
        self.txt = nn.Linear(text_dim, d)
        self.txt_pos = nn.Parameter(torch.randn(1, max_text, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.two_frame = two_frame
        if two_frame:  # also sees the start frame, marked by a learned frame embedding
            self.frame = nn.Parameter(torch.randn(1, 2, 1, d) * 0.02)
        self.body = _encoder(d, layers, heads)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, z, text, text_pad, z0=None):
        """z (B,256,384) final frame; text (B,L,text_dim) contextual token embeddings, text_pad (B,L) True
        at padding; z0 the start frame (two_frame only)."""
        B, L = text_pad.shape
        img = self.inp(z) + self.pos
        if self.two_frame:
            img = torch.cat([self.inp(z0) + self.pos + self.frame[:, 0], img + self.frame[:, 1]], 1)
        h = torch.cat([self.cls.expand(B, -1, -1), self.txt(text) + self.txt_pos[:, :L], img], 1)
        pad = torch.cat([text_pad.new_zeros(B, 1), text_pad, text_pad.new_zeros(B, img.shape[1])], 1)
        return self.head(self.body(h, src_key_padding_mask=pad)[:, 0]).squeeze(-1)


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


class DirectTokenClassifier(nn.Module):
    """Start patch tokens + one push + instruction TOKEN sequence -> success logit (no future prediction).
    The token-text counterpart of DirectClassifier, with the same text pathway as TokenScorer."""

    def __init__(self, text_dim=TEXT_DIM, d=256, layers=10, heads=8, max_text=64, dropout=0.0, token_drop=0.0):
        super().__init__()
        self.inp = nn.Linear(FEAT_DIM, d)
        self.pos = nn.Parameter(torch.randn(1, N_TOK, d) * 0.02)
        self.act = _mlp(ACT_DIM, d, d)
        self.txt = nn.Linear(text_dim, d)
        self.txt_pos = nn.Parameter(torch.randn(1, max_text, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.body = _encoder(d, layers, heads, dropout)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.token_drop = token_drop  # training only: hide this fraction of image patches (augmentation)

    def forward(self, z, a, text, text_pad):
        B, L = text_pad.shape
        a = self.act(a * ACT_SCALE).unsqueeze(1)
        h = torch.cat([self.cls.expand(B, -1, -1), a, self.txt(text) + self.txt_pos[:, :L], self.inp(z) + self.pos], 1)
        img_pad = text_pad.new_zeros(B, N_TOK)
        if self.training and self.token_drop > 0:
            img_pad = torch.rand(B, N_TOK, device=z.device) < self.token_drop
        pad = torch.cat([text_pad.new_zeros(B, 2), text_pad, img_pad], 1)
        return self.head(self.body(h, src_key_padding_mask=pad)[:, 0]).squeeze(-1)
