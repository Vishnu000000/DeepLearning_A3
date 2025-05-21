import torch
import torch.nn as nn
import random
from typing import Tuple, Optional, Union, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RecurrentConfig:
    """Configuration for recurrent neural network layers."""
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    cell_type: str
    bidirectional: bool = False

class RecurrentCellFactory:
    """Factory for creating different types of recurrent neural network cells."""
    
    @staticmethod
    def create_cell(config: RecurrentConfig) -> nn.Module:
        """Create a recurrent neural network cell based on configuration.
        
        Args:
            config: Configuration for the recurrent cell
            
        Returns:
            An instance of the specified recurrent cell type
            
        Raises:
            ValueError: If cell_type is invalid
        """
        if config.cell_type == "LSTM":
            return nn.LSTM(
                config.input_size, 
                config.hidden_size, 
                config.num_layers, 
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        elif config.cell_type == "GRU":
            return nn.GRU(
                config.input_size, 
                config.hidden_size, 
                config.num_layers, 
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        elif config.cell_type == "RNN":
            return nn.RNN(
                config.input_size, 
                config.hidden_size, 
                config.num_layers, 
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        else:
            raise ValueError(f"Invalid cell type: {config.cell_type}")

class Encoder(nn.Module):
    """Encoder module for sequence-to-sequence models.
    
    This module processes input sequences and produces hidden states
    that capture the semantic information of the input.
    """
    
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, 
                 num_layers: int = 1, cell_type: str = "RNN", dropout: float = 0.0,
                 bidirectional: bool = False):
        """Initialize the encoder.
        
        Args:
            input_size: Size of input vocabulary
            embedding_size: Dimension of character embeddings
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers
            cell_type: Type of recurrent cell (LSTM, GRU, or RNN)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
            
        Raises:
            ValueError: If any of the parameters are invalid
        """
        super().__init__()
        
        if input_size <= 0 or embedding_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("Invalid dimensions for encoder")
        if dropout < 0 or dropout >= 1:
            raise ValueError("Dropout must be between 0 and 1")
            
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout if num_layers > 1 else 0.0)
        
        # Recurrent layer
        config = RecurrentConfig(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
            bidirectional=bidirectional
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
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            # Apply embedding and dropout
            embedded = self.dropout(self.embedding(input))
            
            # Process through recurrent layer
            if self.cell_type == 'LSTM':
                outputs, (hidden, cell) = self.recurrent_layer(embedded)
                return outputs, (hidden, cell)
            else:
                outputs, hidden = self.recurrent_layer(embedded)
                return outputs, hidden
                
        except Exception as e:
            logger.error(f"Error in encoder forward pass: {str(e)}")
            raise RuntimeError(f"Encoder forward pass failed: {str(e)}")

class Decoder(nn.Module):
    """Decoder module for sequence-to-sequence models.
    
    This module generates output sequences based on the encoder's
    hidden states and previous decoder outputs.
    """
    
    def __init__(self, output_size: int, embedding_size: int, hidden_size: int, 
                 num_layers: int = 1, cell_type: str = "RNN", dropout: float = 0.0):
        """Initialize the decoder.
        
        Args:
            output_size: Size of output vocabulary
            embedding_size: Dimension of character embeddings
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers
            cell_type: Type of recurrent cell (LSTM, GRU, or RNN)
            dropout: Dropout probability
            
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
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout if num_layers > 1 else 0.0)
        
        # Recurrent layer
        config = RecurrentConfig(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type
        )
        
        self.recurrent_layer = RecurrentCellFactory.create_cell(config)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Process single timestep through the decoder.
        
        Args:
            input: Input token tensor of shape (batch_size,)
            hidden: Hidden state from previous timestep
            
        Returns:
            Tuple containing:
            - Output prediction tensor of shape (batch_size, output_size)
            - Hidden state (and cell state for LSTM)
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            # Prepare input
            input = input.unsqueeze(1)  # Add sequence length dimension
            embedded = self.dropout(self.embedding(input))
            
            # Process through recurrent layer
            if self.cell_type == "LSTM":
                hidden, cell = hidden
                outputs, (hidden, cell) = self.recurrent_layer(embedded, (hidden, cell))
                hidden = (hidden, cell)
            else:
                outputs, hidden = self.recurrent_layer(embedded, hidden)
            
            # Generate prediction
            outputs = outputs.squeeze(1)
            prediction = self.fc_out(outputs)
            
            return prediction, hidden
            
        except Exception as e:
            logger.error(f"Error in decoder forward pass: {str(e)}")
            raise RuntimeError(f"Decoder forward pass failed: {str(e)}")

class Seq2Seq(nn.Module):
    """Sequence-to-sequence model for transliteration.
    
    This model combines an encoder and decoder to perform
    character-level sequence transduction.
    """
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        """Initialize the sequence-to-sequence model.
        
        Args:
            encoder: Encoder module
            decoder: Decoder module
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
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """Process a batch of sequences through the model.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Output predictions tensor of shape (batch_size, tgt_len, output_size)
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            batch_size = src.shape[0]
            tgt_len = tgt.shape[1]
            tgt_vocab_size = self.decoder.output_size
            
            # Initialize output tensor
            outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
            
            # Encode source sequence
            if self.encoder.cell_type == 'LSTM':
                _, (hidden, cell) = self.encoder(src)
                decoder_hidden = (hidden, cell)
            else:
                _, hidden = self.encoder(src)
                decoder_hidden = hidden
            
            # Initialize decoder input with start token
            decoder_input = tgt[:, 0]
            
            # Decode sequence
            for t in range(1, tgt_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t] = decoder_output
                
                # Teacher forcing
                top = decoder_output.argmax(1)
                decoder_input = tgt[:, t] if random.random() < teacher_forcing_ratio else top
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in seq2seq forward pass: {str(e)}")
            raise RuntimeError(f"Seq2Seq forward pass failed: {str(e)}")

    def inference(self, src: torch.Tensor, max_len: int, 
                 sos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        """Generate output sequence for given input sequence.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            max_len: Maximum length of output sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            
        Returns:
            Output predictions tensor of shape (batch_size, max_len, output_size)
            
        Raises:
            RuntimeError: If inference fails
        """
        try:
            batch_size = src.shape[0]
            tgt_vocab_size = self.decoder.output_size
            
            # Initialize output tensor
            outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)
            
            # Encode source sequence
            if self.encoder.cell_type == 'LSTM':
                _, (hidden, cell) = self.encoder(src)
                decoder_hidden = (hidden, cell)
            else:
                _, hidden = self.encoder(src)
                decoder_hidden = hidden
            
            # Initialize decoder input with start token
            decoder_input = torch.tensor([sos_idx] * batch_size, device=self.device)
            
            # Decode sequence
            for t in range(max_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t] = decoder_output
                
                # Get next token
                decoder_input = decoder_output.argmax(1)
                
                # Check if all sequences have ended
                if (decoder_input == eos_idx).all():
                    break
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in seq2seq inference: {str(e)}")
            raise RuntimeError(f"Seq2Seq inference failed: {str(e)}")