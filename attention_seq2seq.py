import torch
import torch.nn as nn

class SeqEncoder(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hid_dim, 
                 layers=1, rnn_type='lstm', drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_dim, emb_dim)
        self.rnn = getattr(nn, rnn_type.upper())(
            emb_dim, hid_dim, layers,
            dropout=drop if layers>1 else 0,
            batch_first=True
        )
        
    def encode(self, x):
        emb = self.emb(x)
        return self.rnn(emb)

class FocusMechanism(nn.Module):
    def __init__(self, hid_dim, mode='dot', drop=0.1):
        super().__init__()
        self.mode = mode
        if mode == 'general':
            self.proj = nn.Linear(hid_dim, hid_dim)
        elif mode == 'concat':
            self.proj = nn.Linear(hid_dim*2, hid_dim)
            self.v = nn.Parameter(torch.rand(hid_dim))
        
        self.drop = nn.Dropout(drop)
        
    def compute(self, dec_hid, enc_outs, mask=None):
        if self.mode == 'dot':
            scores = torch.bmm(enc_outs, dec_hid.unsqueeze(2)).squeeze(2)
        elif self.mode == 'general':
            scores = torch.bmm(self.proj(enc_outs), dec_hid.unsqueeze(2)).squeeze(2)
        elif self.mode == 'concat':
            expanded = dec_hid.unsqueeze(1).expand_as(enc_outs)
            combined = torch.tanh(self.proj(torch.cat((enc_outs, expanded), 2)))
            scores = torch.matmul(combined, self.v)
            
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
            
        attn = self.drop(torch.softmax(scores, 1))
        ctx = torch.bmm(attn.unsqueeze(1), enc_outs).squeeze(1)
        return attn, ctx

class AttnDecoder(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hid_dim, 
                 layers=1, rnn_type='lstm', drop=0.3, 
                 attn_mode='dot'):
        super().__init__()
        self.emb = nn.Embedding(vocab_dim, emb_dim)
        self.attn = FocusMechanism(hid_dim, attn_mode, drop)
        self.rnn = getattr(nn, rnn_type.upper())(
            emb_dim + hid_dim, hid_dim, layers,
            dropout=drop if layers>1 else 0,
            batch_first=True
        )
        self.out = nn.Linear(hid_dim*2, vocab_dim)
        
    def step(self, x, hid, enc_outs, mask=None):
        emb = self.emb(x.unsqueeze(1))
        attn, ctx = self.attn.compute(hid[-1], enc_outs, mask)
        rnn_in = torch.cat([emb, ctx.unsqueeze(1)], 2)
        out, new_hid = self.rnn(rnn_in, hid)
        return self.out(torch.cat([out.squeeze(1), ctx], 1)), new_hid, attn

class AttnSeqModel(nn.Module):
    def __init__(self, enc, dec, device):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.dev = device
        
    def process(self, src, tgt=None, max_len=50, teach_prob=0.5):
        outputs = []
        attns = []
        
        enc_outs, hid = self.enc.encode(src)
        dec_in = src[:,0] if tgt is None else tgt[:,0]
        
        for t in range(max_len if tgt is None else tgt.size(1)):
            pred, hid, attn = self.dec.step(dec_in, hid, enc_outs)
            outputs.append(pred)
            attns.append(attn)
            
            if tgt is not None and random.random() < teach_prob:
                dec_in = tgt[:,t]
            else:
                dec_in = pred.argmax(1)
                
            if (dec_in == 2).all() and tgt is None: break
                
        return torch.stack(outputs, 1), torch.stack(attns, 1)