import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers.attention import MultiHeadAttention
from layers.feedforward import FeedForward

'''
TODO:
- mask <pad> tokens (encoder and decoder)
'''
class Transformer(nn.Module):
    def __init__(self, N, d_model, d_k, d_v, h, d_ff, d_token):
        super(Transformer, self).__init__()
        self.N = N
        self.d_model, self.d_k, self.d_v, self.h, self.d_ff, self.d_token = d_model, d_k, d_v, h, d_ff, d_token

        self.dropout = nn.Dropout(p=0.1)
        self.encoders = [TransformerEncoder(d_model, d_k, d_v, h, d_ff) for i in range(N)]
        self.decoders = [TransformerDecoder(d_model, d_k, d_v, h, d_ff) for i in range(N)]

        self.embeddings_and_linear = nn.Linear(d_token, d_model, bias=False) #Shared matrix for embedding the in-/ouput tokens and pre-softmax layer

    def forward(self, inputs, outputs):
        assert len(inputs)==len(outputs) #Padded to same length
        assert inputs.size(-1)==outputs.size(-1)==d_token

        batch_size, n, _ = inputs.size()
        pe = create_positional_encoding(n, self.d_model)

        #Multiply weights ✕ sqrt(d_model), add positional embedding
        inputs  = self.dropout(np.sqrt(self.d_model)*self.embeddings_and_linear(inputs) + pe)
        outputs = self.dropout(np.sqrt(self.d_model)*self.embeddings_and_linear(outputs) + pe)

        #TODO: back&forth or  N ✕ enc -> N ✕ dec. ?!
        for i in range(self.N):
            inputs = self.encoders[i](inputs)
        for i in range(self.N):
            outputs = self.decoders[i](outputs, inputs)

        #Apply linear transposed
        outputs = outputs @ self.embeddings_and_linear.weight
        probabilities = F.softmax(outputs, dim=-1)

        assert probabilities.size(-1)==d_token and probabilities.size(-2)==n
        return probabilities
        
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff):
        super(TransformerEncoder, self).__init__()
        self.d_model, self.d_k, self.d_v, self.h, self.d_ff = d_model, d_k, d_v, h, d_ff
        
        self.dropout = nn.Dropout(p=0.1)
        
        self.multihead_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.multihead_attention_ln = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff)
        self.ff_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x_attention = self.dropout(self.multihead_attention(x, x, x))
        x_attention_normed = self.multihead_attention_ln(x_attention + x)
        x_ff = self.dropout(self.ff(x_attention_normed))
        x_ff_normed = self.ff_ln(x_ff + x_attention_normed)
        return x_ff_normed

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff):
        super(TransformerDecoder, self).__init__()
        self.d_model, self.d_k, self.d_v, self.h, self.d_ff = d_model, d_k, d_v, h, d_ff
        
        self.dropout = nn.Dropout(p=0.1)
        
        self.masked_multihead_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.masked_multihead_attention_ln = nn.LayerNorm(d_model)
        
        self.multihead_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.multihead_attention_ln = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model, d_ff)
        self.ff_ln = nn.LayerNorm(d_model)

    def forward(self, x, out_encoder):
        #Apply masked self-attention
        x_masked_attention = self.dropout(self.masked_multihead_attention(x, x, x, apply_mask=True))
        x_masked_attention_normed = self.masked_multihead_attention_ln(x_masked_attention + x)
        
        #Apply cross-attention
        #TODO/CARE: which input order? In the paper Figure 2, the order is now V,K,Q -> x_masked_attention is Q
        x_attention = self.multihead_attention(x_masked_attention_normed, out_encoder, out_encoder)
        x_attention_normed = self.multihead_attention_ln(x_attention + x_masked_attention_normed)

        #Apply feed forward
        x_ff = self.ff(x_attention_normed)
        x_ff_normed = self.ff_ln(x_ff + x_attention_normed)
        return x_ff_normed

def create_positional_encoding(n,d):
    pe_pos = np.arange(n).reshape((n,1))
    pe_dim = np.arange(d).reshape((1,d))
    pe = pe_pos / (np.power(1e4, pe_dim//d)) #TODO/CARE: use integer division or not?
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return torch.unsqueeze(torch.from_numpy(pe.astype(np.float32)), dim=0)

if __name__ == "__main__":
    batch_size = 2
    d_model, d_k, d_v, h, d_ff, d_token = 64, 8, 8, 8, 256, 16

    transformer = Transformer(2, d_model, d_k, d_v, h, d_ff, d_token)
    inputs = torch.rand(batch_size, 10, d_token)
    outputs = torch.rand(batch_size, 10, d_token)

    probabilities = transformer(inputs, outputs)
    print(probabilities.shape)