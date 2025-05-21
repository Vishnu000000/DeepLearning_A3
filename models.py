import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the neural transliteration model"""
    input_dim: int
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.2
    attention_heads: int = 4
    use_layer_norm: bool = True
    use_residual: bool = True
    max_sequence_length: int = 100
    use_convolution: bool = True
    use_transformer: bool = True
    use_lstm: bool = True

class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder with dynamic feature extraction"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Character embedding with position encoding
        self.char_embedding = nn.Embedding(config.input_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_sequence_length)
        
        # Multi-scale feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(config.hidden_dim, config.hidden_dim // 2, kernel_size=k),
                nn.BatchNorm1d(config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ) for k in [2, 3, 4]
        ])
        
        # Feature fusion with attention
        self.fusion_attention = nn.MultiheadAttention(
            config.hidden_dim // 2,
            num_heads=config.attention_heads,
            dropout=config.dropout
        )
        
        # Bidirectional processing
        self.forward_lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim // 2,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        self.backward_lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim // 2,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection with gating
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Character embedding with position encoding
        embedded = self.char_embedding(x)
        embedded = self.pos_encoding(embedded)
        
        # Multi-scale feature extraction
        conv_features = []
        for conv_layer in self.conv_layers:
            # Reshape for convolution
            conv_input = embedded.transpose(1, 2)
            # Apply convolution
            conv_output = conv_layer(conv_input)
            # Reshape back
            conv_features.append(conv_output.transpose(1, 2))
        
        # Feature fusion with attention
        fused_features = torch.stack(conv_features, dim=0)
        fused_features, _ = self.fusion_attention(
            fused_features, fused_features, fused_features
        )
        fused_features = fused_features.mean(dim=0)
        
        # Bidirectional processing
        forward_out, (forward_h, forward_c) = self.forward_lstm(fused_features)
        backward_out, (backward_h, backward_c) = self.backward_lstm(fused_features.flip(1))
        
        # Combine forward and backward
        combined_out = torch.cat([forward_out, backward_out], dim=-1)
        combined_h = torch.cat([forward_h, backward_h], dim=-1)
        combined_c = torch.cat([forward_c, backward_c], dim=-1)
        
        # Output projection with gating
        projected = self.output_projection(combined_out)
        gate = self.gate(combined_out)
        output = projected * gate + combined_out * (1 - gate)
        
        return output, (combined_h, combined_c)

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence processing"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class AdaptiveAttention(nn.Module):
    """Adaptive attention with dynamic scoring"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_dim // config.attention_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Scoring functions
        self.scoring_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity()
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(config.dropout)
    
    def _compute_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using multiple scoring functions"""
        # Reshape for multi-head processing
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.config.attention_heads, self.head_dim)
        key = key.view(batch_size, -1, self.config.attention_heads, self.head_dim)
        
        # Compute different scoring functions
        dot_score = torch.matmul(query, key.transpose(-2, -1))
        cosine_score = F.cosine_similarity(query.unsqueeze(-2), key.unsqueeze(-3), dim=-1)
        euclidean_score = -torch.cdist(query, key)
        
        # Combine scores with learned weights
        weights = F.softmax(self.scoring_weights, dim=0)
        scores = (weights[0] * dot_score + 
                 weights[1] * cosine_score + 
                 weights[2] * euclidean_score)
        
        return scores
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Compute attention scores
        scores = self._compute_scores(q, k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Compute weighted sum
        context = torch.matmul(attention_weights, v)
        
        # Project output
        output = self.out_proj(context)
        
        # Apply residual connection and layer normalization
        if self.config.use_residual:
            output = self.layer_norm(output + query)
        else:
            output = self.layer_norm(output)
        
        return output, attention_weights

class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder with adaptive attention"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Character embedding with position encoding
        self.char_embedding = nn.Embedding(config.output_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_sequence_length)
        
        # Adaptive attention
        self.self_attention = AdaptiveAttention(config)
        self.cross_attention = AdaptiveAttention(config)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            config.hidden_dim * 2,  # Input + attention context
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection with gating
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, 
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_hidden: Tuple[torch.Tensor, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Character embedding with position encoding
        embedded = self.char_embedding(x)
        embedded = self.pos_encoding(embedded)
        
        # Self-attention
        self_attn_out, _ = self.self_attention(embedded, embedded, embedded)
        
        # Cross-attention
        cross_attn_out, attention_weights = self.cross_attention(
            self_attn_out, encoder_output, encoder_output, mask
        )
        
        # Combine attention outputs
        combined = torch.cat([self_attn_out, cross_attn_out], dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined, encoder_hidden)
        
        # Output projection with gating
        projected = self.output_projection(lstm_out)
        gate = self.gate(lstm_out)
        output = projected * gate + lstm_out * (1 - gate)
        
        return output, attention_weights

class NeuralTransliterator(nn.Module):
    """Neural transliteration model with multi-scale processing"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = MultiScaleEncoder(config)
        
        # Decoder
        self.decoder = MultiScaleDecoder(config)
    
    def forward(self, 
                source: torch.Tensor,
                target: torch.Tensor,
                source_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        encoder_output, encoder_hidden = self.encoder(source)
        
        # Decode
        decoder_output, attention_weights = self.decoder(
            target, encoder_output, encoder_hidden, source_mask
        )
        
        return decoder_output, attention_weights
    
    def generate(self, 
                source: torch.Tensor,
                max_length: int,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate output sequence using beam search"""
        batch_size = source.size(0)
        
        # Encode
        encoder_output, encoder_hidden = self.encoder(source)
        
        # Initialize decoder input
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        # Generate
        outputs = []
        attention_weights = []
        
        for _ in range(max_length):
            decoder_output, attention = self.decoder(
                decoder_input, encoder_output, encoder_hidden
            )
            
            # Get most likely token
            decoder_input = decoder_output.argmax(dim=-1)
            
            outputs.append(decoder_output)
            attention_weights.append(attention)
        
        return torch.cat(outputs, dim=1), torch.cat(attention_weights, dim=1) 