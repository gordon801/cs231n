import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Initialise list of indices across each column (i) (i.e. 0, 1, ..., max_len-1)
        i = torch.arange(0, max_len).reshape(-1,1)
        
        # Initialise list of exponents across each row (j) (i.e. 0, 2, 4, ..., embed_dim-2)
        pe_term = 10000 ** (-(torch.arange(0, embed_dim, 2).float()/embed_dim))
        
        # For all rows, select every second column starting from 0 (j) (i.e. 0, 2, 4...)
        pe[:,:,0::2] = torch.sin(i * pe_term)
        
        # For all rows, select every second column starting from 1 (j) (i.e. 1, 3, 5...)
        pe[:,:,1::2] = torch.cos(i * pe_term)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Output = X + P
        # Add the appropriate (unique) values from the positional encoding array (i.e. 
        # until the end of the sequence, measured by sequence length S)
        output = x + self.pe[:,:S]
        
        # Apply dropout afterwards
        output = self.dropout(output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Split shape of query, key, and value vectors into (N, T, H, E/H)
        # Swap axes 1 and 2 to match dimensions in hint.
        H = self.n_head
        Q = self.query(query).view(N, S, H, E//H).transpose(1,2)
        K = self.key(key).view(N, T, H, E//H).transpose(1,2)
        V = self.value(value).view(N, T, H, E//H).transpose(1,2)

        # Calculate similarities by matrix multiplying query and key, scale by 1/(E/H)**0.5
        # E = Q*K.T (N, H, T, E/H) * (N, H, E/H, T) = (N, H, T, T)
        similarity_scores = torch.matmul(Q, K.transpose(2,3)) / (E/H)**0.5
        
        # Apply attention masking to similarity scores via attn_mask (S, T)
        if attn_mask is not None:
            similarity_scores = similarity_scores.masked_fill(attn_mask==0, float("-inf")) 

        # Compute attention weights by applying softmax along dim=-1 to normalise the
        # scores for each query over all keys.
        attn_weights = F.softmax(similarity_scores, dim=-1)
        
        # Apply dropout to attention weights output. 
        attn_weights = self.attn_drop(attn_weights)
        
        # Compute output Y (N, H, T, E//H) = A (N, H, T, T) * V (N, H, T, E//H)
        output = torch.matmul(attn_weights, V)

        # Reorder axes into original shape, reshape output into (N, S, E), then project output.
        output = self.proj(output.transpose(1,2).reshape(N, S, E))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


