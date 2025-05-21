import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelParams:
    def __init__(self, vocab_size, target_size):
        self.embed_size = 256
        self.latent_size = 512
        self.depth = 3
        self.drop_prob = 0.2
        self.head_count = 4
        self.max_seq = 100
        self.enable_skip = True
        self.use_norm = True

class HierarchicalEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.embed = nn.Embedding(params.vocab_size, params.embed_size)
        self.pos_embed = PositionEmbedder(params.embed_size, params.max_seq)
        
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(params.embed_size, params.embed_size//2, k),
                nn.GroupNorm(1, params.embed_size//2),
                nn.GELU(),
                nn.Dropout(params.drop_prob)
            ) for k in [2,3,5]
        ])
        
        self.feature_merger = nn.MultiheadAttention(
            params.embed_size//2, params.head_count, params.drop_prob
        )
        
        self.seq_processor = nn.LSTM(
            params.embed_size, params.latent_size//2,
            num_layers=params.depth,
            dropout=params.drop_prob,
            bidirectional=True,
            batch_first=True
        )
        
        self.transform = nn.Sequential(
            nn.Linear(params.latent_size, params.latent_size),
            nn.LayerNorm(params.latent_size) if params.use_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout(params.drop_prob)
        )
        self.merge_gate = nn.Linear(params.latent_size, params.latent_size)

    def process(self, x):
        emb = self.embed(x)
        emb = self.pos_embed(emb)
        
        conv_features = []
        for extractor in self.feature_extractors:
            feat = extractor(emb.permute(0,2,1)).permute(0,2,1)
            conv_features.append(feat)
        
        merged, _ = self.feature_merger(
            torch.stack(conv_features), 
            torch.stack(conv_features),
            torch.stack(conv_features)
        )
        merged = merged.mean(0)
        
        processed, (h, c) = self.seq_processor(merged)
        
        transformed = self.transform(processed)
        gate = torch.sigmoid(self.merge_gate(processed))
        return transformed*gate + processed*(1-gate), (h,c)

class PositionEmbedder(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2)*(-math.log(100)/dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[...,0::2] = torch.sin(pos*div)
        pe[...,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class DynamicFocus(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim = params.latent_size
        self.heads = params.head_count
        self.head_dim = self.dim // self.heads
        
        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)
        
        self.score_weights = nn.Parameter(torch.ones(3))
        self.norm = nn.LayerNorm(self.dim) if params.use_norm else nn.Identity()
        self.drop = nn.Dropout(params.drop_prob)

    def compute_attention(self, q, k, v, mask=None):
        b, t, _ = q.size()
        q = q.view(b, t, self.heads, self.head_dim)
        k = k.view(b, t, self.heads, self.head_dim)
        
        dot = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        cos = F.cosine_similarity(q.unsqueeze(-2), k.unsqueeze(-3), dim=-1)
        dist = -torch.cdist(q, k)
        
        weights = F.softmax(self.score_weights, 0)
        scores = weights[0]*dot + weights[1]*cos + weights[2]*dist
        
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
            
        attn = F.softmax(scores, -1)
        attn = self.drop(attn)
        
        output = (attn @ v).reshape(b, t, self.dim)
        return self.out(output), attn

    def forward(self, q, k, v, mask=None):
        q_proj = self.query(q)
        k_proj = self.key(k)
        v_proj = self.value(v)
        
        result, attn = self.compute_attention(q_proj, k_proj, v_proj, mask)
        if self.params.enable_skip:
            result = result + q
        return self.norm(result), attn

class SequenceGenerator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.embed = nn.Embedding(params.target_size, params.embed_size)
        self.pos_embed = PositionEmbedder(params.embed_size, params.max_seq)
        
        self.self_focus = DynamicFocus(params)
        self.cross_focus = DynamicFocus(params)
        
        self.seq_cell = nn.LSTM(
            params.embed_size + params.latent_size,
            params.latent_size,
            num_layers=params.depth,
            dropout=params.drop_prob,
            batch_first=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(params.latent_size, params.latent_size),
            nn.LayerNorm(params.latent_size) if params.use_norm else nn.Identity(),
            nn.ELU(),
            nn.Dropout(params.drop_prob),
            nn.Linear(params.latent_size, params.target_size)
        )

    def decode_step(self, x, mem, hidden, mask=None):
        emb = self.pos_embed(self.embed(x))
        self_attn, _ = self.self_focus(emb, emb, emb)
        cross_attn, weights = self.cross_focus(self_attn, mem, mem, mask)
        combined = torch.cat([self_attn, cross_attn], -1)
        
        out, hidden = self.seq_cell(combined, hidden)
        return self.output_layer(out), weights, hidden

class CharTranslator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = HierarchicalEncoder(params)
        self.decoder = SequenceGenerator(params)
        self.params = params

    def forward(self, src, tgt, src_mask=None):
        enc_out, enc_hidden = self.encoder.process(src)
        dec_out, attn_weights, _ = self.decoder.decode_step(tgt, enc_out, enc_hidden, src_mask)
        return dec_out, attn_weights

    def predict(self, src, max_len, device):
        batch_size = src.size(0)
        enc_out, enc_hidden = self.encoder.process(src)
        
        tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        outputs = []
        
        for _ in range(max_len):
            logits, attn, enc_hidden = self.decoder.decode_step(
                tokens[:, -1:], enc_out, enc_hidden)
            tokens = torch.cat([tokens, logits.argmax(-1)], -1)
            outputs.append(logits)
            
        return torch.cat(outputs, 1), tokens