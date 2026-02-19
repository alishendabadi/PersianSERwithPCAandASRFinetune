import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class MultiHeadAttentivePoolinig(nn.Module):
    def __init__(self, d_attention , d_model, n_heads, dropout):
        super().__init__()

        # assume d_v = d_attention 
        self.d_attention  = d_attention 
        self.n_heads = n_heads
        self.dropout = dropout

        self.attention_proj = nn.Linear(in_features=d_model, out_features=d_attention  * n_heads)
        self.score_proj = nn.Linear(in_features=d_attention  * n_heads, out_features=1 * n_heads)
        self.value_proj = nn.Linear(in_features=d_model, out_features=d_attention  * n_heads)
        ###___self.query_proj = nn.Linear(d_model, d_attention  * n_heads)

        #final Linear Layer
        self.finalFC = nn.Linear(d_attention  * n_heads, d_model)

    def forward(self, k, v, scale=True):
        ###___q = self.query_proj(q) # batch * Seq * (hd_attention )
        k = self.score_proj(F.tanh(F.dropout(self.attention_proj(k), p=self.dropout))) # batch * Seq * (h)
        v = F.dropout(self.value_proj(v), p=self.dropout) # batch * Seq * (hd_attention )

        N = k.shape[0] #batch size in batch_first mode
        T = k.shape[1] #Sequence length


        #change the shape to:
        # (N, T, h, d_attention ) --> (N, h, T, d_attention ) in order to matmul to work correctly with dims
        ###___q = q.view(N, T, self.n_heads, self.d_attention ).transpose(1, 2)
        k = k.view(N, T, self.n_heads, 1).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_attention ).transpose(1, 2)

        #Compute attention weights: (N,h,T,dk)@(N,h,dk,T) = (N,h,T,T)
        ###___att_scores = q @ k.transpose(-2, -1) / sqrt(self.d_attention )
        if scale:
            att_scores = k.transpose(-2, -1) / sqrt(self.d_attention)
        elif not scale:
            att_scores = k.transpose(-2, -1)
        att_weights = F.softmax(att_scores, dim=-1)

        #(N,h,T,T) @ (N,h,T,dk) = (N,h,T,dk)
        A = att_weights @ v

        A = A.contiguous().view(N, self.d_attention  * self.n_heads) #(N, T, h*dk)

        A = self.finalFC(A) #(N * d_model)

        return A, att_weights
    
class MultiHeadQKVAttention(nn.Module):
    def __init__(self, d_k , d_model, n_heads, dropout):
        super().__init__()

        self.d_k  = d_k 
        self.n_heads = n_heads
        self.dropout = dropout

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)

        self.fc = nn.Linear(d_k * n_heads, d_model)

    def forward(self, q, k, v):
        q = F.dropout(self.query(q), p=self.dropout)
        k = F.dropout(self.key(k), p=self.dropout)
        v = F.dropout(self.value(v), p=self.dropout)

        N = k.shape[0]
        T = k.shape[1]

        q = q.view(N, 1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_score = F.tanh(q @ k.transpose(-2, -1) / sqrt(self.d_k))
        attn_weights = F.softmax(attn_score, dim=-1)

        A = attn_weights @ v
        A = A.transpose(1, 2)
        A = A.contiguous().view(N, self.d_k * self.n_heads)
        A = self.fc(A)

        return A, attn_weights
    

# one set of projections
# class MultiHeadAttentionWithoutQ(nn.Module):
#     def __init__(self, d_attention , d_model, n_heads, dropout):
#         super().__init__()

#         # assume d_v = d_attention 
#         self.d_attention  = d_attention 
#         self.n_heads = n_heads
#         self.dropout = dropout

#         self.attention_proj = nn.Linear(in_features=d_model, out_features=d_attention  * n_heads)
#         self.score_proj = nn.Linear(in_features=d_attention  * n_heads, out_features=1 * n_heads)
#         #self.value_proj = nn.Linear(in_features=d_model, out_features=d_attention  * n_heads)
#         ###___self.query_proj = nn.Linear(d_model, d_attention  * n_heads)

#         #final Linear Layer
#         self.finalFC = nn.Linear(d_attention  * n_heads, d_model)

#     def forward(self, k, v):
#         ###___q = self.query_proj(q) # batch * Seq * (hd_attention )
#         v = F.dropout(self.attention_proj(k), p=self.dropout)
#         k = self.score_proj(F.tanh(v)) # batch * Seq * (h)

#         N = k.shape[0] #batch size in batch_first mode
#         T = k.shape[1] #Sequence length


#         #change the shape to:
#         # (N, T, h, d_attention ) --> (N, h, T, d_attention ) in order to matmul to work correctly with dims
#         ###___q = q.view(N, T, self.n_heads, self.d_attention ).transpose(1, 2)
#         k = k.view(N, T, self.n_heads, 1).transpose(1, 2)
#         v = v.view(N, T, self.n_heads, self.d_attention ).transpose(1, 2)

#         #Compute attention weights: (N,h,T,dk)@(N,h,dk,T) = (N,h,T,T)
#         ###___att_scores = q @ k.transpose(-2, -1) / sqrt(self.d_attention )
#         att_scores = k.transpose(-2, -1) / sqrt(self.d_attention )
#         att_weights = F.softmax(att_scores, dim=-1)

#         #(N,h,T,T) @ (N,h,T,dk) = (N,h,T,dk)
#         A = att_weights @ v

#         A = A.contiguous().view(N, self.d_attention  * self.n_heads) #(N, T, h*dk)

#         A = self.finalFC(A) #(N * d_model)

#         return A, att_weights