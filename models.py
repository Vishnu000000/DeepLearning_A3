import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelConfig:
    """Configuration class for model parameters"""
    def __init__(self, input_size, output_size, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.embed_dim = kwargs.get('embed_dim', 256)
        self.hidden_dim = kwargs.get('hidden_dim', 512)
        self.num_layers = kwargs.get('num_layers', 2)
        self.num_heads = kwargs.get('num_heads', 4)
        self.dropout = kwargs.get('dropout', 0.3)
        self.rnn_type = kwargs.get('rnn_type', 'lstm')
        self.bidirectional = kwargs.get('bidirectional', True)

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    def __init__(self, dim, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention scoring
        self.score_weights = nn.Parameter(torch.ones(3))
        self.dropout = nn.Dropout(dropout)
        
    def compute_attention(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        dot_product = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        cosine_sim = F.cosine_similarity(q.unsqueeze(-2), k.unsqueeze(-3), dim=-1)
        distance = -torch.cdist(q, k)
        
        # Combine scores
        weights = F.softmax(self.score_weights, 0)
        scores = weights[0] * dot_product + weights[1] * cosine_sim + weights[2] * distance
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        
        # Compute attention weights
        attention = F.softmax(scores, -1)
        attention = self.dropout(attention)
        
        # Compute output
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.output_proj(output), attention

class NeuralEncoder(nn.Module):
    """Neural encoder with attention"""
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.input_size, config.embed_dim)
        self.position_encoding = PositionalEncoding(config.embed_dim)
        
        # RNN layer
        rnn_class = getattr(nn, config.rnn_type.upper())
        self.rnn = rnn_class(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        
        # Output processing
        self.output_proj = nn.Linear(
            config.hidden_dim * (2 if config.bidirectional else 1),
            config.hidden_dim
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x):
        # Embed and add positional encoding
        embedded = self.embedding(x)
        embedded = self.position_encoding(embedded)
        
        # Process through RNN
        outputs, hidden = self.rnn(embedded)
        
        # Project and normalize
        outputs = self.output_proj(outputs)
        outputs = self.layer_norm(outputs)
        
        return outputs, hidden

class NeuralDecoder(nn.Module):
    """Neural decoder with attention"""
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.output_size, config.embed_dim)
        self.position_encoding = PositionalEncoding(config.embed_dim)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            config.hidden_dim,
            config.num_heads,
            config.dropout
        )
        
        # RNN layer
        rnn_class = getattr(nn, config.rnn_type.upper())
        self.rnn = rnn_class(
            input_size=config.embed_dim + config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output processing
        self.output_processor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_size)
        )
        
    def decode_step(self, x, memory, hidden, mask=None):
        # Embed and add positional encoding
        embedded = self.embedding(x)
        embedded = self.position_encoding(embedded)
        
        # Compute attention
        context, attention_weights = self.attention.compute_attention(
            hidden[-1].unsqueeze(1), memory, memory, mask
        )
        
        # Combine features
        combined = torch.cat([embedded, context], -1)
        
        # Process through RNN
        output, new_hidden = self.rnn(combined, hidden)
        
        # Generate output
        output = self.output_processor(torch.cat([output, context], -1))
        
        return output, attention_weights, new_hidden

class NeuralTranslator(nn.Module):
    """Neural sequence-to-sequence translator"""
    def __init__(self, config):
        super().__init__()
        self.encoder = NeuralEncoder(config)
        self.decoder = NeuralDecoder(config)
        self.config = config
        
    def forward(self, source, target, source_mask=None):
        # Encode source sequence
        encoded, encoder_hidden = self.encoder(source)
        
        # Decode target sequence
        decoded, attention_weights, _ = self.decoder.decode_step(
            target, encoded, encoder_hidden, source_mask
        )
        
        return decoded, attention_weights
    
    def predict(self, source, max_length, device):
        batch_size = source.size(0)
        
        # Encode source sequence
        encoded, encoder_hidden = self.encoder(source)
        
        # Initialize generation
        tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        outputs = []
        
        # Generate sequence
        for _ in range(max_length):
            # Decode step
            logits, attention, encoder_hidden = self.decoder.decode_step(
                tokens[:, -1:], encoded, encoder_hidden
            )
            
            # Get next token
            next_token = logits.argmax(-1)
            tokens = torch.cat([tokens, next_token], -1)
            outputs.append(logits)
            
            # Check for end token
            if (next_token == 2).all():
                break
        
        return torch.cat(outputs, 1), tokens