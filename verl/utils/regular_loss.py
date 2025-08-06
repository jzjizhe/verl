from multiprocessing import process
import torch
import torch.nn.functional as F
import math

def process_hidden(hidden_states_ls):
    return hidden_states_ls[0]

def compute_golden_loss(hidden_states_ls, golden_hidden_ls, hidden_mask, golden_mask,token_level_scores,align_type,normalize=False):
    # scores=token_level_scores.sum(dim=1)
    # flip_scores=1-scores.unsqueeze(1)
    # hidden_length = hidden_states_ls[0].size(1)
    # golden_length = golden_hidden_ls[0].size(1)
    # hidden_mask=hidden_mask[:,-hidden_length:]
    # golden_mask=golden_mask[:,-golden_length:]
    h1 = process_hidden(hidden_states_ls)  # (bsz, seq_len, hidden_size)
    h2 = process_hidden(golden_hidden_ls)  # (bsz, seq_len, hidden_size)
    if align_type=="last_token":
        last_token_indices=hidden_mask.sum(dim=1)-1
        batch_indices = torch.arange(h1.size(0), device=h1.device)
        h1 = h1[batch_indices, last_token_indices] # (bsz, hidden_size)
        last_golden_indices=golden_mask.sum(dim=1)-1
        golden_batch_indices = torch.arange(h2.size(0), device=h2.device)
        h2 = h2[golden_batch_indices, last_golden_indices] # (bsz, hidden_size) 
    elif align_type=="all2last":
        last_golden_indices = golden_mask.sum(dim=1) - 1
        golden_batch_indices = torch.arange(h2.size(0), device=h2.device)
        h2 = h2[golden_batch_indices, last_golden_indices].unsqueeze(1)  # (bsz, 1, hidden_size)
    else:
        raise ValueError(f"Invalid alignment type: {align_type}")
    if normalize:
        h1 = F.normalize(h1,p=2, dim=-1)
        h2 = F.normalize(h2,p=2, dim=-1)
    cos_sim = F.cosine_similarity(h1, h2, dim=-1)
    # cos_sim = F.cosine_similarity(h1, h2, dim=-1)*flip_scores
    hidden_golden_loss = 1 - cos_sim.mean()
    return hidden_golden_loss

# def compute_golden_loss(hidden_states_ls, golden_hidden_ls, hidden_mask, golden_mask, 
#                        use_multilayer=False, alignment_strategy='original', temperature=0.1,normalize=False):
#     """
#     改进的golden loss计算函数
    
#     Args:
#         hidden_states_ls: 生成文本的hidden states列表 [layer_num, bsz, seq_len, hidden_size]
#         golden_hidden_ls: 黄金文本的hidden states列表 [layer_num, bsz, seq_len, hidden_size]
#         hidden_mask: 生成文本的mask [bsz, seq_len]
#         golden_mask: 黄金文本的mask [bsz, seq_len]
#         use_multilayer: 是否使用多层hidden states
#         alignment_strategy: 对齐策略 ('attention', 'global', 'min_length')
#         temperature: 注意力温度参数
#     """
#     if use_multilayer:
#         # 使用多层hidden states，加权平均
#         layer_weights = [0.1, 0.2, 0.3, 0.4]  # 可以调整权重
#         total_loss = 0
        
#         for i, (hidden_states, golden_hidden) in enumerate(zip(hidden_states_ls, golden_hidden_ls)):
#             if i < len(layer_weights):
#                 layer_loss = compute_single_layer_loss(
#                     hidden_states, golden_hidden, hidden_mask, golden_mask, 
#                     alignment_strategy, temperature
#                 )
#                 total_loss += layer_weights[i] * layer_loss
        
#         return total_loss
#     else:
#         # 只使用第一层
#         return compute_single_layer_loss(
#             hidden_states_ls, golden_hidden_ls, hidden_mask, golden_mask,
#             alignment_strategy, temperature,normalize
#         )

# def compute_single_layer_loss(hidden_states, golden_hidden, hidden_mask, golden_mask,
#                             alignment_strategy='original', temperature=0.1,normalize=False):
#     """
#     单层hidden states的loss计算
#     """
#     # bsz, hidden_length, hidden_size = hidden_states.shape
#     # _, golden_length, _ = golden_hidden.shape
    
#     # # 处理mask
#     # hidden_mask = hidden_mask[:, -hidden_length:]
#     # golden_mask = golden_mask[:, -golden_length:]
    
#     if alignment_strategy == 'attention':
#         return attention_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask, temperature)
#     elif alignment_strategy == 'global':
#         return global_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask)
#     elif alignment_strategy == 'min_length':
#         return min_length_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask)
#     elif alignment_strategy == 'original':
#         return original_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask,normalize=normalize)
#     else:
#         raise ValueError(f"Invalid alignment strategy: {alignment_strategy}")

# def attention_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask, temperature=0.1):
#     """
#     使用注意力机制的对齐损失
#     """
#     # 归一化
#     hidden_norm = F.normalize(hidden_states, dim=-1)
#     golden_norm = F.normalize(golden_hidden, dim=-1)
    
#     # 计算注意力权重
#     attention_weights = torch.matmul(hidden_norm, golden_norm.transpose(-2, -1))
#     attention_weights = attention_weights / math.sqrt(hidden_states.size(-1))
    
#     # 应用mask
#     hidden_mask_expanded = hidden_mask.unsqueeze(-1)  # [bsz, seq_len, 1]
#     golden_mask_expanded = golden_mask.unsqueeze(1)   # [bsz, 1, seq_len]
#     mask = hidden_mask_expanded & golden_mask_expanded  # [bsz, seq_len, seq_len]
    
#     # 在mask位置应用softmax
#     attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
#     attention_weights = F.softmax(attention_weights / temperature, dim=-1)
    
#     # 对齐golden hidden states
#     aligned_golden = torch.matmul(attention_weights, golden_norm)
    
#     # 计算相似度
#     similarity = torch.cosine_similarity(hidden_norm, aligned_golden, dim=-1)
    
#     # 只对有效位置计算loss
#     valid_mask = hidden_mask.bool()
#     masked_similarity = similarity * valid_mask.float()
#     valid_count = valid_mask.sum()
    
#     if valid_count > 0:
#         loss = 1 - (masked_similarity.sum() / valid_count)
#     else:
#         loss = torch.tensor(0.0, device=hidden_states.device)
    
#     return loss

# def global_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask):
#     """
#     全局对齐损失
#     """
#     # 全局平均池化（只考虑有效位置）
#     hidden_pooled = (hidden_states * hidden_mask.unsqueeze(-1)).sum(dim=1) / hidden_mask.sum(dim=1, keepdim=True)
#     golden_pooled = (golden_hidden * golden_mask.unsqueeze(-1)).sum(dim=1) / golden_mask.sum(dim=1, keepdim=True)
    
#     # 归一化
#     hidden_norm = F.normalize(hidden_pooled, dim=-1)
#     golden_norm = F.normalize(golden_pooled, dim=-1)
    
#     # 计算全局相似度
#     similarity = torch.cosine_similarity(hidden_norm, golden_norm, dim=-1)
#     loss = 1 - similarity.mean()
    
#     return loss

# def min_length_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask):
#     """
#     最小长度对齐损失
#     """
#     bsz = hidden_states.shape[0]
#     total_loss = 0
    
#     for i in range(bsz):
#         # 获取有效长度
#         hidden_len = hidden_mask[i].sum().item()
#         golden_len = golden_mask[i].sum().item()
#         min_len = min(hidden_len, golden_len)
        
#         if min_len == 0:
#             continue
        
#         # 截取到最小长度
#         hidden_valid = hidden_states[i, :min_len, :]
#         golden_valid = golden_hidden[i, :min_len, :]
        
#         # 归一化
#         hidden_norm = F.normalize(hidden_valid, dim=-1)
#         golden_norm = F.normalize(golden_valid, dim=-1)
        
#         # 计算相似度
#         similarity = torch.cosine_similarity(hidden_norm, golden_norm, dim=-1)
#         total_loss += 1 - similarity.mean()
    
#     return total_loss / bsz if bsz > 0 else torch.tensor(0.0, device=hidden_states.device)

# def original_alignment_loss(hidden_states, golden_hidden, hidden_mask, golden_mask,normalize=False):
#     """
#     原始的loss计算方法（保持兼容性），加入归一化
#     """
#     hidden_length = hidden_states[0].size(1)
#     golden_length = golden_hidden[0].size(1)
#     hidden_mask = hidden_mask[:, -hidden_length:]
#     golden_mask = golden_mask[:, -golden_length:]
    
#     h1 = process_hidden(hidden_states)  # (bsz, seq_len, hidden_size)
#     h2 = process_hidden(golden_hidden)  # (bsz, seq_len, hidden_size)
    
#     # mask: (bsz, seq_len)
#     valid_mask = hidden_mask.bool() & golden_mask.bool()  # 只对都有效的位置
#     h1_valid = h1[valid_mask]
#     h2_valid = h2[valid_mask]
    
#     if h1_valid.size(0) == 0:
#         return torch.tensor(0.0, device=h1_valid.device)
    
#     if normalize:
#         # 加入归一化
#         h1_valid = F.normalize(h1_valid, dim=-1)
#         h2_valid = F.normalize(h2_valid, dim=-1)
    
#     cos_sim = F.cosine_similarity(h1_valid, h2_valid, dim=-1)
#     hidden_golden_loss = 1 - cos_sim.mean()
    
#     return hidden_golden_loss

def compute_infonce_loss(hidden_states_ls, mask: torch.Tensor = None, temperature: float = 0.1):
    """
    计算InfoNCE损失
    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        mask: (batch_size, seq_len) - 可选，用于mask padding tokens
        temperature: float, 温度参数
    Returns:
        infonce_loss: scalar tensor
        infonce_accuracy: scalar tensor
    """
    hidden_states=process_hidden(hidden_states_ls)
    batch_size, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device
    if mask is not None:
        valid_tokens = mask.sum(dim=1)  # (batch_size,)
        valid_hidden_states = []
        for i in range(batch_size):
            valid_len = valid_tokens[i].item()
            if valid_len > 0:
                valid_hidden_states.append(hidden_states[i, :valid_len])
        if not valid_hidden_states:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        all_hidden_states = torch.cat(valid_hidden_states, dim=0)  # (total_valid_tokens, hidden_size)
        total_valid_tokens = all_hidden_states.size(0)
    else:
        all_hidden_states = hidden_states.view(-1, hidden_size)
        total_valid_tokens = all_hidden_states.size(0)
    # 归一化
    normalized_hidden_states = F.normalize(all_hidden_states, p=2, dim=1)
    similarity_matrix = torch.matmul(normalized_hidden_states, normalized_hidden_states.T) / temperature
    labels = torch.arange(total_valid_tokens, device=device)
    infonce_loss = F.cross_entropy(similarity_matrix, labels)
    predictions = similarity_matrix.argmax(dim=1)
    accuracy = (predictions == labels).float().mean()
    return infonce_loss, accuracy 