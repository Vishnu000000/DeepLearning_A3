import torch
from torch import nn

class EncoderCore(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, 
                 n_layers=1, cell='lstm', dropout=0.5, 
                 bidirectional=False):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = self._make_rnn(
            emb_dim, hid_dim, n_layers, 
            cell, dropout, bidirectional
        )
        
    def _make_rnn(self, in_dim, hid_dim, layers, 
                cell_type, drop, bidir):
        cell = cell_type.lower()
        if cell == 'lstm':
            return nn.LSTM(in_dim, hid_dim, layers,
                          dropout=drop if layers>1 else 0,
                          bidirectional=bidir,
                          batch_first=True)
        elif cell == 'gru':
            return nn.GRU(in_dim, hid_dim, layers,
                         dropout=drop if layers>1 else 0,
                         bidirectional=bidir,
                         batch_first=True)
        return nn.RNN(in_dim, hid_dim, layers,
                     dropout=drop if layers>1 else 0,
                     bidirectional=bidir,
                     batch_first=True)
    
    def forward(self, src):
        embedded = self.embed(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class DecoderCore(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, 
                 n_layers=1, cell='lstm', dropout=0.5):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = self._make_rnn(emb_dim, hid_dim, 
                                layers=n_layers, 
                                cell_type=cell, 
                                drop=dropout)
        self.out = nn.Linear(hid_dim, vocab_size)
        
    def _make_rnn(self, in_dim, hid_dim, layers, 
                cell_type, drop):
        cell = cell_type.lower()
        if cell == 'lstm':
            return nn.LSTM(in_dim, hid_dim, layers,
                          dropout=drop if layers>1 else 0,
                          batch_first=True)
        elif cell == 'gru':
            return nn.GRU(in_dim, hid_dim, layers,
                         dropout=drop if layers>1 else 0,
                         batch_first=True)
        return nn.RNN(in_dim, hid_dim, layers,
                     dropout=drop if layers>1 else 0,
                     batch_first=True)
    
    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        emb = self.embed(x)
        output, hidden = self.rnn(emb, hidden)
        return self.out(output.squeeze(1)), hidden

class SeqMapper(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.dev = device
        
    def translate(self, src, trg, teacher_ratio=0.5):
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        vocab_size = self.dec.out.out_features
        
        results = torch.zeros(batch_size, max_len, 
                            vocab_size).to(self.dev)
        
        _, hid = self.enc(src)
        dec_in = trg[:,0]
        
        for t in range(1, max_len):
            pred, hid = self.dec(dec_in, hid)
            results[:,t] = pred
            dec_in = trg[:,t] if random.random() < teacher_ratio else pred.argmax(1)
            
        return results
    
    def generate(self, src, max_len, sos=1, eos=2):
        batch_size = src.shape[0]
        vocab_size = self.dec.out.out_features
        
        output = torch.zeros(batch_size, max_len, 
                           vocab_size).to(self.dev)
        
        _, hid = self.enc(src)
        dec_in = torch.full((batch_size,), sos, 
                          device=self.dev)
        
        for t in range(max_len):
            pred, hid = self.dec(dec_in, hid)
            output[:,t] = pred
            dec_in = pred.argmax(1)
            if (dec_in == eos).all(): break
            
        return output