import torch.nn as nn
import torch.nn.functional as F
import torch
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, mlp_layers=2):
        super().__init__()
        self.layers = self._build_mlp(input_size, hidden_size, mlp_layers)

    def _build_mlp(self, input_size, hidden_size, mlp_layers):
        layers = []
        for i in range(mlp_layers):
            in_dim = input_size if i == 0 else hidden_size
            out_dim = input_size if i == mlp_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < mlp_layers - 1:
                layers.append(nn.GELU())
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 可学习的查询向量（语义探测器）
        self.query = nn.Parameter(torch.randn(hidden_dim))  # 形状: [dim]
        
    def forward(self, hidden_states, mask=None):
        """
        输入: 
            hidden_states - 序列的hidden state，形状: [batch_size, seq_len, dim]
            mask - padding掩码，1表示有效token，0表示padding，形状: [batch_size, seq_len]
        输出: 
            全局向量，形状: [batch_size, dim]
        """
        # 计算每个token与查询向量的相似度（内积）
        # [batch_size, seq_len, dim] · [dim] → [batch_size, seq_len]
        scores = torch.matmul(hidden_states, self.query)
        
        # 处理掩码：将padding位置的分数设为极小值，确保softmax后权重接近0
        if mask is not None:
            # 掩码形状转为 [batch_size, seq_len]，与scores一致
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重（softmax归一化）
        attn_weights = F.softmax(scores, dim=1)  # 形状: [batch_size, seq_len]
        
        # 加权聚合：[batch_size, seq_len] × [batch_size, seq_len, dim] → [batch_size, dim]
        # 先扩展权重维度: [batch_size, seq_len] → [batch_size, 1, seq_len]
        attn_weights = attn_weights.unsqueeze(1)
        # 矩阵乘法: [batch_size, 1, seq_len] × [batch_size, seq_len, dim] → [batch_size, 1, dim]
        pooled = torch.bmm(attn_weights, hidden_states).squeeze(1)
        
        return pooled



def add_mlp(input_size, hidden_size, mlp_layers=2):
    """返回一个继承 nn.Module 的 MLP 实例"""
    return MLP(input_size, hidden_size, mlp_layers)

def add_attention_pooling(token_dim):
    return AttentionPooling(token_dim)

