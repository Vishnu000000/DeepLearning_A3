import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder with dynamic feature extraction"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Multi-scale feature extraction
        self.scale_embeddings = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim // 2),
            nn.Linear(input_dim, hidden_dim // 4),
            nn.Linear(input_dim, hidden_dim // 8)
        ])
        
        # Dynamic feature fusion
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Bidirectional processing
        self.forward_gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.backward_gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-scale feature extraction
        scale_features = []
        for scale_embed in self.scale_embeddings:
            scale_features.append(scale_embed(x))
        
        # Dynamic feature fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = sum(w * f for w, f in zip(weights, scale_features))
        
        # Bidirectional processing
        forward_out, forward_hidden = self.forward_gru(fused_features)
        backward_out, backward_hidden = self.backward_gru(fused_features.flip(1))
        
        # Combine forward and backward
        combined_out = torch.cat([forward_out, backward_out], dim=-1)
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
        
        return combined_out, combined_hidden

class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism with dynamic scoring"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Dynamic scoring components
        self.query_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.key_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Adaptive scoring
        self.scoring_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transform query and key
        query = self.query_transform(query)
        key = self.key_transform(key)
        
        # Multiple scoring functions
        dot_score = torch.matmul(query, key.transpose(-2, -1))
        cosine_score = F.cosine_similarity(query.unsqueeze(-2), key.unsqueeze(-3), dim=-1)
        euclidean_score = -torch.cdist(query, key)
        
        # Combine scores with learned weights
        weights = F.softmax(self.scoring_weights, dim=0)
        scores = (weights[0] * dot_score + 
                 weights[1] * cosine_score + 
                 weights[2] * euclidean_score)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, value)
        
        return context, attention_weights

class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder with adaptive attention"""
    def __init__(self, 
                 output_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Multi-scale processing
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Linear(hidden_dim, hidden_dim // 8)
        ])
        
        # Adaptive attention
        self.attention = AdaptiveAttention(hidden_dim)
        
        # Decoder GRU
        self.gru = nn.GRU(
            hidden_dim + output_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_hidden: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-scale processing
        scale_features = []
        for scale_proj in self.scale_projections:
            scale_features.append(scale_proj(encoder_hidden[-1].unsqueeze(1)))
        
        # Combine scale features
        combined_features = torch.cat(scale_features, dim=-1)
        
        # Attention
        context, attention_weights = self.attention(
            combined_features,
            encoder_output,
            encoder_output,
            mask
        )
        
        # Decode
        decoder_input = torch.cat([x, context], dim=-1)
        output, _ = self.gru(decoder_input)
        
        # Project to output
        output = self.output(output)
        
        return output, attention_weights

class MultiScaleSeq2Seq(nn.Module):
    """Multi-scale sequence-to-sequence model"""
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = MultiScaleEncoder(
            input_dim, hidden_dim, num_layers, dropout
        )
        self.decoder = MultiScaleDecoder(
            output_dim, hidden_dim, num_layers, dropout
        )
        
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