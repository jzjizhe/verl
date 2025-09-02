from multiprocessing import process
import torch
import torch.nn.functional as F
import math
import pandas as pd
import random
from fastdtw import fastdtw
import numpy as np
def off_diagonal(x):
    """获取矩阵的非对角线元素"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

def vicreg_h1(h1, h2,flip_score):
    layers = h1.shape[1]
    total_loss = 0.0
    
    for i in range(layers):
        z_a = h1[:, i, :]  # 生成特征（第i层）
        z_b = h2[:, i, :]  # 黄金特征（第i层）
        batch_size, embed_dim = z_a.shape[0], z_a.shape[1]
        # 1. 不变性损失：生成特征与黄金特征对齐（两者都参与）
        sim_loss = F.mse_loss(z_a, z_b,reduction="none").mean(dim=-1)
        sim_loss=(sim_loss*flip_score).mean()
        
        # 2. 方差损失：仅对生成特征h1施加约束（保证多样性）
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a))  # 只计算z_a的方差损失
        
        # 3. 协方差损失：仅对生成特征h1施加约束（减少冗余）
        z_a_centered = z_a - z_a.mean(dim=0)
        cov_z_a = (z_a_centered.T @ z_a_centered) / (batch_size - 1)
        cov_loss = torch.pow(off_diagonal(cov_z_a), 2).sum() / embed_dim  # 只计算z_a的协方差损失
        
        # 累加当前层的损失
        layer_loss = (25* sim_loss + 
                     25 * std_loss + 
                     1 * cov_loss)
        total_loss += layer_loss
    total_loss /= layers
    return total_loss
    

def vicreg_loss(h1, h2,flip_score):
    layers=h1.shape[1]
    loss=0
    for i in range(layers):
        z_a=h1[:,i,:]
        z_b=h2[:,i,:]
        batch_size, embed_dim = z_a.shape[0], z_a.shape[1]
        # 不变性损失 (Invariance Loss)
        sim_loss = F.mse_loss(z_a, z_b,reduction="none").mean(dim=-1)
        sim_loss=(sim_loss*flip_score).mean()
        
        # 方差损失 (Variance Loss)
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)  # 每个维度的标准差
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = (torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))) * 0.5
        
        # 协方差损失 (Covariance Loss)
        # 中心化特征
        z_a_centered = z_a - z_a.mean(dim=0)
        z_b_centered = z_b - z_b.mean(dim=0)
        # 计算协方差矩阵
        cov_z_a = (z_a_centered.T @ z_a_centered) / (batch_size - 1)
        cov_z_b = (z_b_centered.T @ z_b_centered) / (batch_size - 1)
        # 计算非对角线元素的平方和
        cov_loss = (torch.pow(off_diagonal(cov_z_a), 2).sum() + 
                    torch.pow(off_diagonal(cov_z_b), 2).sum()) / embed_dim
        
        # 总损失
        layer_loss=25* sim_loss + 25 * std_loss + 1 * cov_loss
        loss += layer_loss
    loss /= layers
    return loss
def process_hidden(hidden_states_ls):
    # return hidden_states_ls[0]
    return hidden_states_ls

def compute_golden_loss(hidden_states_ls, golden_hidden_ls, hidden_mask, golden_mask,token_level_scores,projector,align_type,loss_type,normalize,uid,config=None):
    h1 = hidden_states_ls  # (bsz, layers, seq_len, hidden_dim)
    h2 = golden_hidden_ls  # (bsz, layers, seq_len, hidden_dim)
    if config.get("add_mlp",False) or config.get("add_attention_pooling",False):
        # 将h1的数据类型转成和projector一致
        # print(h1.dtype)
        # print(projector.layers[0].weight.dtype)
        h1=projector(h1[:,0,:,:])
    if config.get("add_mlp_golden",False) or config.get("add_attention_pooling",False):
        # h2 = h2.to(next(projector.parameters()).dtype)
        with torch.no_grad():
            h2=projector(h2[:,0,:,:])
    if normalize:
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
    if loss_type=="cosine":
        cos_sim = F.cosine_similarity(h1, h2, dim=-1)
        hidden_golden_loss = 1 - cos_sim.mean()
    elif loss_type=="l1_wrong":
        hidden_golden_loss = F.l1_loss(h1, h2,reduction="none")
        hidden_golden_loss=hidden_golden_loss.mean(dim=-1)
        flip_score=1-token_level_scores.sum(-1)
        # total_flip=flip_score.sum()
        # hidden_golden_loss = (
        #     (hidden_golden_loss * flip_score).sum() / total_flip 
        #     if total_flip > 0 
        #     else torch.zeros_like(hidden_golden_loss).sum()  # 返回同设备/类型的零张量
        # )
        hidden_golden_loss =(hidden_golden_loss * flip_score).mean()
    elif loss_type=="l1":
        hidden_golden_loss = F.l1_loss(h1, h2)
    elif loss_type=="mse":
        hidden_golden_loss = F.mse_loss(h1, h2)
    elif loss_type=="cosine_wrong":
        cos_sim = F.cosine_similarity(h1, h2, dim=-1).mean(dim=-1)
        flip_score = 1-token_level_scores.sum(-1)
        # total_flip = flip_score.sum()
        # hidden_golden_loss = (
        #     1 - (cos_sim * flip_score).sum() / total_flip 
        #     if total_flip > 0 
        #     else torch.zeros_like(cos_sim).sum()  # 返回同设备/类型的零张量
        # ) 
        hidden_golden_loss = 1 - (cos_sim * flip_score).mean()
    elif loss_type=="contrastive":
        df = pd.DataFrame({"uid": uid, "original_idx": range(len(uid))})
        unique_df = df.drop_duplicates("uid", keep="first").reset_index(drop=True)  # 重置索引为0,1,2,...
        uid_to_unique_idx = {uid: idx for idx, uid in enumerate(unique_df["uid"])}
        labels = [uid_to_unique_idx[u] for u in uid]
        labels = torch.tensor(labels, device=h1.device)
        h2 = h2[unique_df["original_idx"].values]  # 用原始索引提取数据
        temperature = 0.1
        logits = torch.matmul(h1, h2.t())/temperature
        hidden_golden_loss = F.cross_entropy(logits, labels)
    elif loss_type=="contra_all":
        hidden_golden_loss=contra_all(h1,h2,uid,temperature=0.1)
    elif loss_type=="contra_all_wrong":
        hidden_golden_loss=contra_all_wrong(h1,h2,uid,token_level_scores,temperature=0.1)
    elif loss_type=="vicreg":
        flip_score=1-token_level_scores.sum(-1)
        hidden_golden_loss=vicreg_loss(h1,h2,flip_score)
    elif loss_type=="vicreg_h1":
        flip_score=1-token_level_scores.sum(-1)
        hidden_golden_loss=vicreg_h1(h1,h2,flip_score)
    elif loss_type=="attention":
        hidden_golden_loss=cross_attention_loss(h1[:,0,:,:],h2[:,0,:,:],hidden_mask,golden_mask,temperature=0.1)
    elif loss_type=="dtw_cosine":
        hidden_golden_loss=dtw_loss(h1[:,0,:,:],h2[:,0,:,:],hidden_mask,golden_mask,radius=50,dist_metric='cosine')
    elif loss_type=="dtw_gpu_cosine":
        hidden_golden_loss=dtw_gpu_loss(h1[:,0,:,:],h2[:,0,:,:],hidden_mask,golden_mask,dist_metric='cosine')
    elif loss_type=="dtw_e":
        hidden_golden_loss=dtw_loss(h1[:,0,:,:],h2[:,0,:,:],radius=50,dist_metric='euclidean')
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return hidden_golden_loss


def contra_all(A, B, uids, temperature=0.1):
    """
    A: (bsz_A, dim)  
    B: (bsz_B, dim)  
    uids: list[str], 长度bsz_B，标识B中每行的UID
    """
    # 1. 对B去重
    bsz_A=A.size(0)
    df = pd.DataFrame({"uid": uids, "original_idx": range(len(uids))})
    unique_df = df.drop_duplicates("uid", keep="first").reset_index(drop=True)
    B_unique = B[unique_df["original_idx"].values]  # (M, dim), M <= bsz_B
    
    # 2. 建立UID到B_unique索引的映射
    uid_to_idx = {uid: idx for idx, uid in enumerate(unique_df["uid"])}
    
    # 3. 合并所有负样本池：[A, B_unique]
    neg_pool = torch.cat([A, B_unique], dim=0)  # (bsz_A + M, dim)
    
    # 4. 计算相似度矩阵
    logits = torch.matmul(A, neg_pool.T) / temperature  # (bsz_A, bsz_A + M)
    
    # 5. 构建正样本位置（A[i]对应的B_unique[j]在neg_pool中的位置是bsz_A + j）
    labels = torch.tensor(
        [bsz_A + uid_to_idx[uid] for uid in uids], 
        device=A.device
    )
    
    # 6. 屏蔽无效位置（A[i]不应与A[i]自己计算相似度）
    mask = torch.eye(bsz_A, dtype=torch.bool, device=A.device)
    logits[:, :bsz_A][mask] = -float('inf')  # 将A的对角线设为-inf
    
    # 7. 计算Loss
    loss = F.cross_entropy(logits, labels)
    return loss

def contra_all_wrong(A, B, uids,scores, temperature=0.1):
    """
    A: (bsz_A, dim)  
    B: (bsz_B, dim)  
    uids: list[str], 长度bsz_B，标识B中每行的UID
    """
    idx=scores[:,-1]==0
    if not idx.any():  # 如果没有满足条件的样本
        return torch.zeros_like(A).sum()
    A=A[idx]
    B=B[idx]
    uids=uids[idx.cpu().numpy()]
    # 1. 对B去重
    bsz_A=A.size(0)
    df = pd.DataFrame({"uid": uids, "original_idx": range(len(uids))})
    unique_df = df.drop_duplicates("uid", keep="first").reset_index(drop=True)
    B_unique = B[unique_df["original_idx"].values]  # (M, dim), M <= bsz_B
    
    # 2. 建立UID到B_unique索引的映射
    uid_to_idx = {uid: idx for idx, uid in enumerate(unique_df["uid"])}
    
    # 3. 合并所有负样本池：[A, B_unique]
    neg_pool = torch.cat([A, B_unique], dim=0)  # (bsz_A + M, dim)
    
    # 4. 计算相似度矩阵
    logits = torch.matmul(A, neg_pool.T) / temperature  # (bsz_A, bsz_A + M)
    
    # 5. 构建正样本位置（A[i]对应的B_unique[j]在neg_pool中的位置是bsz_A + j）
    labels = torch.tensor(
        [bsz_A + uid_to_idx[uid] for uid in uids], 
        device=A.device
    )

    
    # 6. 屏蔽无效位置（A[i]不应与A[i]自己计算相似度）
    mask = torch.eye(bsz_A, dtype=torch.bool, device=A.device)
    logits[:, :bsz_A][mask] = -float('inf')  # 将A的对角线设为-inf
    
    # 7. 计算Loss
    loss = F.cross_entropy(logits, labels)
    return loss

def cross_attention_loss(gen_hidden, gold_hidden, gen_mask=None, gold_mask=None, temperature=0.1):
    """
    支持padding和mask的交叉对齐损失函数，使用余弦相似度损失
    
    参数:
        gen_hidden: 生成序列的hidden state，形状为[batch, gen_seq_len, dim]
        gold_hidden: 黄金答案的hidden state，形状为[batch, gold_seq_len, dim]
        gen_mask: 生成序列的mask，1表示有效token，0表示padding，形状为[batch, gen_seq_len]
                  若为None则默认所有位置都是有效token
        gold_mask: 黄金答案的mask，1表示有效token，0表示padding，形状为[batch, gold_seq_len]
                  若为None则默认所有位置都是有效token
        temperature: 注意力缩放因子，控制注意力分布的陡峭程度
    
    返回:
        loss: 交叉对齐损失值
    """
    batch_size, gen_seq_len, dim = gen_hidden.shape
    gold_seq_len = gold_hidden.shape[1]
    if gen_mask is None:
        gen_mask = torch.ones(batch_size, gen_seq_len, device=gen_hidden.device)
    if gold_mask is None:
        gold_mask = torch.ones(batch_size, gold_seq_len, device=gold_hidden.device)
    
    # 计算注意力分数: [batch, gen_seq_len, gold_seq_len]
    attn_scores = torch.bmm(gen_hidden, gold_hidden.transpose(1, 2))  # 内积计算相似度
    attn_scores = attn_scores / (dim **0.5)  # 缩放
    attn_scores = attn_scores / temperature  # 温度调节
    
    # 对padding位置的注意力分数设置为极小值，使其在softmax后权重接近0
    # 将gold_mask扩展为[batch, 1, gold_seq_len]用于广播
    gold_mask_expanded = gold_mask.unsqueeze(1)  # [batch, 1, gold_seq_len]
    attn_scores = attn_scores.masked_fill(gold_mask_expanded == 0, -1e9)
    
    # 计算注意力权重 (对gold序列维度做softmax)
    attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, gen_seq_len, gold_seq_len]
    
    # 生成对齐向量: 基于注意力权重对gold_hidden进行加权求和
    aligned_gold_hidden = torch.bmm(attn_weights, gold_hidden)  # [batch, gen_seq_len, dim]
    
    # 计算生成序列hidden state与对齐向量的余弦相似度损失，忽略padding部分
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(gen_hidden, aligned_gold_hidden, dim=-1)  # [batch, gen_seq_len]
    loss_per_token = (1 - cos_sim) * gen_mask  # [batch, gen_seq_len]
    per_sample_valid_count = gen_mask.sum(dim=1)  # 每个样本的有效token数量 [batch]
    per_sample_loss = loss_per_token.sum(dim=1) / per_sample_valid_count.clamp(min=1e-8)  # 每个样本的平均损失 [batch]
    loss = per_sample_loss.mean()
    return loss


def dtw_loss(gen_hidden, gold_hidden, gen_mask=None, gold_mask=None,radius=50,dist_metric='cosine'):
    """
    参数:
        gen_hidden: 生成序列的hidden state [batch, seq_len, dim]
        gold_hidden: 黄金序列的hidden state [batch, seq_len, dim]
        gen_mask: 生成序列mask [batch, seq_len]
        gold_mask: 黄金序列mask [batch, seq_len]
        
    返回:
        loss: 平均DTW对齐损失
    """
    device = gen_hidden.device
    batch_size, seq_len, dim = gen_hidden.shape
    
    # 处理mask
    if gen_mask is None:
        gen_mask = torch.ones(batch_size, seq_len, device=device)
    if gold_mask is None:
        gold_mask = torch.ones(batch_size, seq_len, device=device)
    
    # 获取有效长度
    gen_lengths = gen_mask.sum(dim=1).int().cpu().numpy()
    gold_lengths = gold_mask.sum(dim=1).int().cpu().numpy()
    
    # 在GPU上计算距离矩阵（保留梯度）
    if dist_metric == 'cosine':
        # 归一化以计算余弦相似度
        gen_norm = F.normalize(gen_hidden, p=2, dim=-1)
        gold_norm = F.normalize(gold_hidden, p=2, dim=-1)
        # 计算余弦相似度矩阵
        cos_sim = torch.bmm(gen_norm, gold_norm.transpose(1, 2))
        full_distance_matrix = 1 - cos_sim  # 转换为距离
    else:  # 欧氏距离
        # 扩展维度以计算成对距离
        gen_exp = gen_hidden.unsqueeze(2)  # [batch, seq_len, 1, dim]
        gold_exp = gold_hidden.unsqueeze(1)  # [batch, 1, seq_len, dim]
        # 计算欧氏距离
        diff = gen_exp - gold_exp
        full_distance_matrix = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)
    
    
    # 移动到CPU以计算最优路径
    # 转换为float32以支持numpy操作
    gen_cpu = gen_hidden.detach().cpu().float().numpy()
    gold_cpu = gold_hidden.detach().cpu().float().numpy()
    
    # 收集所有路径索引
    all_path_indices = []
    
    for b in range(batch_size):
        gen_len = gen_lengths[b]
        gold_len = gold_lengths[b]
        
        if gen_len == 0 or gold_len == 0:
            all_path_indices.append(([], []))
            continue
            
        # 提取有效序列
        gen_seq = gen_cpu[b, :gen_len]
        gold_seq = gold_cpu[b, :gold_len]
        if dist_metric == 'cosine':
            # 使用余弦距离
            def cosine_dist(a, b):
                # 计算余弦距离: 1 - cosine_similarity
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 1.0
                cosine_sim = dot_product / (norm_a * norm_b)
                return 1 - cosine_sim
            _, path = fastdtw(gen_seq, gold_seq, radius=radius, dist=cosine_dist)
        else:
            # 使用欧氏距离
            def euclidean_dist(a, b):
                return np.linalg.norm(a - b)
            _, path = fastdtw(gen_seq, gold_seq, radius=radius, dist=euclidean_dist)
        
        gen_idx = [p[0] for p in path]
        gold_idx = [p[1] for p in path]
        all_path_indices.append((gen_idx, gold_idx))
    
    losses = []
    for b in range(batch_size):
        gen_idx, gold_idx = all_path_indices[b]
        
        if not gen_idx or not gold_idx:
            losses.append(torch.zeros_like(gen_hidden, device=device).sum())
            continue
            
        # 转换索引为张量
        gen_idx_tensor = torch.tensor(gen_idx, dtype=torch.long, device=device)
        gold_idx_tensor = torch.tensor(gold_idx, dtype=torch.long, device=device)
        
        # 从距离矩阵中提取路径上的距离
        path_distances = full_distance_matrix[b, gen_idx_tensor, gold_idx_tensor]
        
        # 计算平均损失
        sample_loss = path_distances.mean()
        losses.append(sample_loss)
    
    # 计算批次平均损失
    if losses:
        total_loss = torch.stack(losses).mean()
    else:
        total_loss = torch.zeros_like(gen_hidden, device=device).sum()
    return total_loss

def dtw_gpu_loss(gen_hidden, gold_hidden, gen_mask=None, gold_mask=None, dist_metric='cosine'):
    """
    优化后的DTW损失计算，适用于最长10个token的短序列
    
    参数:
        gen_hidden: 生成序列的hidden state [batch, seq_len, dim]
        gold_hidden: 黄金序列的hidden state [batch, seq_len, dim]
        gen_mask: 生成序列mask [batch, seq_len]
        gold_mask: 黄金序列mask [batch, seq_len]
        dist_metric: 距离度量方式 ('cosine' 或 'euclidean')
        
    返回:
        loss: 平均DTW对齐损失
    """
    device = gen_hidden.device
    batch_size, gen_seq_len, dim = gen_hidden.shape
    _, gold_seq_len, _ = gold_hidden.shape
    
    # 处理mask（默认全为1）
    if gen_mask is None:
        gen_mask = torch.ones(batch_size, gen_seq_len, device=device)
    if gold_mask is None:
        gold_mask = torch.ones(batch_size, gold_seq_len, device=device)
    
    # 计算距离矩阵 [batch, gen_seq_len, gold_seq_len]
    if dist_metric == 'cosine':
        gen_norm = F.normalize(gen_hidden, p=2, dim=-1)
        gold_norm = F.normalize(gold_hidden, p=2, dim=-1)
        dist_matrix = 1 - torch.bmm(gen_norm, gold_norm.transpose(1, 2))  # 1 - 余弦相似度
    else:  # 欧氏距离
        gen_exp = gen_hidden.unsqueeze(2)  # [batch, gen_len, 1, dim]
        gold_exp = gold_hidden.unsqueeze(1)  # [batch, 1, gold_len, dim]
        dist_matrix = torch.sqrt(torch.sum((gen_exp - gold_exp)**2, dim=-1) + 1e-8)
    
    # 应用mask：将无效位置的距离设为很大的值
    gen_mask_ = gen_mask.unsqueeze(2)  # [batch, gen_len, 1]
    gold_mask_ = gold_mask.unsqueeze(1)  # [batch, 1, gold_len]
    mask = gen_mask_ * gold_mask_  # [batch, gen_len, gold_len]
    dist_matrix = dist_matrix * mask + (1 - mask) * 1e18  # 无效位置距离为极大值
    
    # 初始化DTW累积距离矩阵
    dtw_matrix = torch.full((batch_size, gen_seq_len + 1, gold_seq_len + 1), 
                           float('inf'), device=device)
    dtw_matrix[:, 0, 0] = 0.0  # 起始点
    
    # 填充DTW矩阵（短序列下直接计算完整矩阵效率更高）
    for i in range(1, gen_seq_len + 1):
        for j in range(1, gold_seq_len + 1):
            # 取左、上、左上三个方向的最小值
            min_prev = torch.min(torch.stack([
                dtw_matrix[:, i-1, j],    # 上
                dtw_matrix[:, i, j-1],    # 左
                dtw_matrix[:, i-1, j-1]   # 左上
            ], dim=1), dim=1)[0]
            
            # 当前位置的累积距离
            dtw_matrix[:, i, j] = dist_matrix[:, i-1, j-1] + min_prev
    
    # 计算有效序列长度
    gen_lengths = gen_mask.sum(dim=1).long()
    gold_lengths = gold_mask.sum(dim=1).long()
    
    # 收集每个样本的最终DTW距离（根据有效长度）
    batch_indices = torch.arange(batch_size, device=device)
    final_distances = dtw_matrix[batch_indices, gen_lengths, gold_lengths]
    
    # 计算平均损失（过滤掉无效样本）
    valid_mask = (gen_lengths > 0) & (gold_lengths > 0)
    if valid_mask.any():
        total_loss = final_distances[valid_mask].mean()
    else:
        total_loss = torch.zeros_like(gen_hidden, device=device).sum()
     
    
    return total_loss