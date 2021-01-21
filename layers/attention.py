import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Q: [b, n, d_k]
#K: [k, d_k]
#V: [k, d_v]
#TODO / care: a@b and a.bmm(b) come out the same right?
def scaled_dotproduct_attention(Q,K,V, apply_mask=False):
    assert Q.size(-1)==K.size(-1) and K.size(-2)==V.size(-2)
    batch_size, n, d_k = Q.size()
    k = K.size(-2)

    qk = Q@K.transpose(1,2) # [b, n, k]
    qk /= np.sqrt(d_k) # [b, n, k]

    if apply_mask:
        assert n==k #Only during self attention
        for i in range(n):
            qk[:, i, i+1:] = -np.inf

    weights = F.softmax(qk, dim=-1) # [b, n, k]

    output = weights@V # [b, n, d_v]
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        assert d_model==d_k*h==d_v*h #Not necessarily true, but proposed in the original paper

        self.d_model, self.d_k, self.d_v, self.h = d_model, d_k, d_v, h

        self.Q_projections = nn.ModuleList([nn.Linear(d_model, d_k) for i in range(h)])
        self.K_projections = nn.ModuleList([nn.Linear(d_model, d_k) for i in range(h)])
        self.V_projections = nn.ModuleList([nn.Linear(d_model, d_v) for i in range(h)])

        self.out_projection = nn.Linear(h*d_v, d_model)


    def forward(self, Q, K, V, apply_mask=False):
        Q_projected = [proj(Q) for proj in self.Q_projections]
        K_projected = [proj(K) for proj in self.K_projections]
        V_projected = [proj(V) for proj in self.V_projections]

        attention_outputs = [scaled_dotproduct_attention(Q_projected[i], K_projected[i], V_projected[i], apply_mask) for i in range(self.h)]
        concatenated_outputs = torch.cat(attention_outputs, dim=-1)
        
        output = self.out_projection(concatenated_outputs)
        return output


#Test the layers
if __name__ == "__main__":
    batch_size = 2
    Q, K, V = torch.rand(batch_size, 10,3), torch.rand(batch_size, 5,3), torch.rand(batch_size, 5,2)
    result = scaled_dotproduct_attention(Q,K,V)
    print(result.size())
    print()

    d_model, h = 128, 8
    d_k = d_v = d_model//h
    Q, K, V = torch.rand(batch_size, 10, d_model), torch.rand(batch_size, d_k, d_model), torch.rand(batch_size, d_v, d_model)
    multihead_attention = MultiHeadAttention(d_model, d_k, d_v, h)
    result = multihead_attention(Q,K,V)
    print(result.size())
    print()
