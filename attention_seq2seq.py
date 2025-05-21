import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional, Union, Dict, List
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """Configuration for attention mechanism."""
    hidden_size: int
    attention_type: str = "dot"  # dot, general, concat
    dropout: float = 0.0
    temperature: float = 1.0

@dataclass
class RecurrentConfig:
    """Configuration for recurrent neural network layers."""
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    cell_type: str

class RecurrentCellFactory:
    """Factory for creating different types of recurrent neural network cells."""
    
    @staticmethod
    def create_cell(config: RecurrentConfig) -> nn.Module:
        """Create a recurrent neural network cell based on configuration.
        
        Args:
            config: Configuration for the recurrent cell
            
        Returns:
            An instance of the specified recurrent cell type
        """
        if config.cell_type == "LSTM":
            return nn.LSTM(
                config.input_size, 
                config.hidden_size, 
                config.num_layers, 
                dropout=config.dropout, 
                batch_first=True
            )
        elif config.cell_type == "GRU":
            return nn.GRU(
                config.input_size, 
                config.hidden_size, 
                config.num_layers, 
                dropout=config.dropout, 
                batch_first=True
            )
        else:
            return nn.RNN(
                config.input_size, 
                config.hidden_size, 
                config.num_layers, 
                dropout=config.dropout, 
                batch_first=True
            )

class Encoder(nn.Module):
    """Encoder module for sequence-to-sequence models with attention.
    
    This module processes input sequences and produces hidden states
    that capture the semantic information of the input.
    """
    
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, 
                 num_layers: int = 1, cell_type: str = "RNN", dropout: float = 0.0):
        """Initialize the encoder.
        
        Args:
            input_size: Size of input vocabulary
            embedding_size: Dimension of character embeddings
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers
            cell_type: Type of recurrent cell (LSTM, GRU, or RNN)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout if num_layers > 1 else 0.0)
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        config = RecurrentConfig(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type
        )
        
        self.recurrent_layer = RecurrentCellFactory.create_cell(config)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Process input sequence through the encoder.
        
        Args:
            input: Input sequence tensor of shape (batch_size, seq_len)
            
        Returns:
            Tuple containing:
            - Output tensor of shape (batch_size, seq_len, hidden_size)
            - Hidden state (and cell state for LSTM)
        """
        embedded = self.dropout(self.embedding(input))
        
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.recurrent_layer(embedded)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.recurrent_layer(embedded)
            return outputs, hidden

class Attention(nn.Module):
    """Attention mechanism for sequence-to-sequence models.
    
    This module computes attention weights between decoder hidden states
    and encoder outputs to focus on relevant parts of the input sequence.
    """
    
    def __init__(self, config: AttentionConfig):
        """Initialize the attention mechanism.
        
        Args:
            config: Configuration for attention mechanism
            
        Raises:
            ValueError: If any of the parameters are invalid
        """
        super().__init__()
        
        if config.hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
        if config.dropout < 0 or config.dropout >= 1:
            raise ValueError("Dropout must be between 0 and 1")
        if config.temperature <= 0:
            raise ValueError("Temperature must be positive")
            
        self.hidden_size = config.hidden_size
        self.attention_type = config.attention_type
        self.temperature = config.temperature
        
        # Attention layers based on type
        if self.attention_type == "general":
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.attention_type == "concat":
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))
            
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize parameters
        if self.attention_type == "concat":
            nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vector.
        
        Args:
            hidden: Decoder hidden state of shape (batch_size, hidden_size)
            encoder_outputs: Encoder outputs of shape (batch_size, src_len, hidden_size)
            mask: Optional mask tensor of shape (batch_size, src_len)
            
        Returns:
            Tuple containing:
            - Attention weights of shape (batch_size, src_len)
            - Context vector of shape (batch_size, hidden_size)
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            batch_size = encoder_outputs.shape[0]
            src_len = encoder_outputs.shape[1]
            
            # Repeat hidden state for each source token
            hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
            
            # Calculate attention scores
            if self.attention_type == "dot":
                energy = torch.sum(hidden * encoder_outputs, dim=2)
            elif self.attention_type == "general":
                energy = torch.sum(hidden * self.attention(encoder_outputs), dim=2)
            elif self.attention_type == "concat":
                energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
                energy = torch.sum(self.v * energy, dim=2)
            else:
                raise ValueError(f"Invalid attention type: {self.attention_type}")
            
            # Apply temperature scaling
            energy = energy / self.temperature
            
            # Apply mask if provided
            if mask is not None:
                energy = energy.masked_fill(mask == 0, float('-inf'))
            
            # Calculate attention weights
            attention_weights = F.softmax(energy, dim=1)
            attention_weights = self.dropout(attention_weights)
            
            # Calculate context vector
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            
            return attention_weights, context
            
        except Exception as e:
            logger.error(f"Error in attention forward pass: {str(e)}")
            raise RuntimeError(f"Attention forward pass failed: {str(e)}")

class AttentionDecoder(nn.Module):
    """Decoder with attention mechanism for sequence-to-sequence models.
    
    This module generates output sequences based on encoder hidden states
    and attention context.
    """
    
    def __init__(self, output_size: int, embedding_size: int, hidden_size: int,
                 num_layers: int = 1, cell_type: str = "RNN", dropout: float = 0.0,
                 attention_config: Optional[AttentionConfig] = None):
        """Initialize the attention decoder.
        
        Args:
            output_size: Size of output vocabulary
            embedding_size: Dimension of character embeddings
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers
            cell_type: Type of recurrent cell (LSTM, GRU, or RNN)
            dropout: Dropout probability
            attention_config: Configuration for attention mechanism
            
        Raises:
            ValueError: If any of the parameters are invalid
        """
        super().__init__()
        
        if output_size <= 0 or embedding_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("Invalid dimensions for decoder")
        if dropout < 0 or dropout >= 1:
            raise ValueError("Dropout must be between 0 and 1")
            
        self.output_size = output_size
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout if num_layers > 1 else 0.0)
        
        # Attention mechanism
        if attention_config is None:
            attention_config = AttentionConfig(hidden_size=hidden_size)
        self.attention = Attention(attention_config)
        
        # Recurrent layer
        config = RecurrentConfig(
            input_size=embedding_size + hidden_size,  # Concatenated with context
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type
        )
        
        self.recurrent_layer = RecurrentCellFactory.create_cell(config)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size * 2, output_size)  # Concatenated with context

    def forward(self, input: torch.Tensor, hidden: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
                encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:
        """Process single timestep through the decoder.
        
        Args:
            input: Input token tensor of shape (batch_size,)
            hidden: Hidden state from previous timestep
            encoder_outputs: Encoder outputs of shape (batch_size, src_len, hidden_size)
            mask: Optional mask tensor of shape (batch_size, src_len)
            
        Returns:
            Tuple containing:
            - Output prediction tensor of shape (batch_size, output_size)
            - Hidden state (and cell state for LSTM)
            - Attention weights of shape (batch_size, src_len)
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            # Prepare input
            input = input.unsqueeze(1)  # Add sequence length dimension
            embedded = self.dropout(self.embedding(input))
            
            # Get hidden state for attention
            if self.cell_type == "LSTM":
                hidden_for_attention = hidden[0][-1]  # Last layer's hidden state
            else:
                hidden_for_attention = hidden[-1]  # Last layer's hidden state
            
            # Calculate attention
            attention_weights, context = self.attention(hidden_for_attention, encoder_outputs, mask)
            
            # Concatenate embedded input with context
            rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
            
            # Process through recurrent layer
            if self.cell_type == "LSTM":
                hidden, cell = hidden
                outputs, (hidden, cell) = self.recurrent_layer(rnn_input, (hidden, cell))
                hidden = (hidden, cell)
            else:
                outputs, hidden = self.recurrent_layer(rnn_input, hidden)
            
            # Generate prediction
            outputs = outputs.squeeze(1)
            prediction = self.fc_out(torch.cat((outputs, context), dim=1))
            
            return prediction, hidden, attention_weights
            
        except Exception as e:
            logger.error(f"Error in attention decoder forward pass: {str(e)}")
            raise RuntimeError(f"Attention decoder forward pass failed: {str(e)}")

class Seq2SeqWithAttention(nn.Module):
    """Sequence-to-sequence model with attention mechanism.
    
    This model combines an encoder and attention decoder to perform
    character-level sequence transduction with attention.
    """
    
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder, device: torch.device):
        """Initialize the sequence-to-sequence model with attention.
        
        Args:
            encoder: Encoder module
            decoder: Attention decoder module
            device: Device to run the model on
            
        Raises:
            ValueError: If any of the components are None
        """
        super().__init__()
        
        if encoder is None or decoder is None:
            raise ValueError("Encoder and decoder must not be None")
            
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of sequences through the model.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Tuple containing:
            - Output predictions tensor of shape (batch_size, tgt_len, output_size)
            - Attention weights tensor of shape (batch_size, tgt_len, src_len)
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            batch_size = src.shape[0]
            tgt_len = tgt.shape[1]
            src_len = src.shape[1]
            tgt_vocab_size = self.decoder.output_size
            
            # Initialize output tensors
            outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
            attention_weights = torch.zeros(batch_size, tgt_len, src_len).to(self.device)
            
            # Encode source sequence
            encoder_outputs, hidden = self.encoder(src)
            
            # Initialize decoder input with start token
            decoder_input = tgt[:, 0]
            
            # Decode sequence
            for t in range(1, tgt_len):
                decoder_output, hidden, attn = self.decoder(decoder_input, hidden, encoder_outputs)
                outputs[:, t] = decoder_output
                attention_weights[:, t] = attn
                
                # Teacher forcing
                top = decoder_output.argmax(1)
                decoder_input = tgt[:, t] if random.random() < teacher_forcing_ratio else top
            
            return outputs, attention_weights
            
        except Exception as e:
            logger.error(f"Error in seq2seq with attention forward pass: {str(e)}")
            raise RuntimeError(f"Seq2Seq with attention forward pass failed: {str(e)}")

    def inference(self, src: torch.Tensor, max_len: int,
                 sos_idx: int = 1, eos_idx: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate output sequence for given input sequence.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            max_len: Maximum length of output sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            
        Returns:
            Tuple containing:
            - Output predictions tensor of shape (batch_size, max_len, output_size)
            - Attention weights tensor of shape (batch_size, max_len, src_len)
            
        Raises:
            RuntimeError: If inference fails
        """
        try:
            batch_size = src.shape[0]
            src_len = src.shape[1]
            tgt_vocab_size = self.decoder.output_size
            
            # Initialize output tensors
            outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)
            attention_weights = torch.zeros(batch_size, max_len, src_len).to(self.device)
            
            # Encode source sequence
            encoder_outputs, hidden = self.encoder(src)
            
            # Initialize decoder input with start token
            decoder_input = torch.tensor([sos_idx] * batch_size, device=self.device)
            
            # Decode sequence
            for t in range(max_len):
                decoder_output, hidden, attn = self.decoder(decoder_input, hidden, encoder_outputs)
                outputs[:, t] = decoder_output
                attention_weights[:, t] = attn
                
                # Get next token
                decoder_input = decoder_output.argmax(1)
                
                # Check if all sequences have ended
                if (decoder_input == eos_idx).all():
                    break
            
            return outputs, attention_weights
            
        except Exception as e:
            logger.error(f"Error in seq2seq with attention inference: {str(e)}")
            raise RuntimeError(f"Seq2Seq with attention inference failed: {str(e)}")
