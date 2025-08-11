from multiprocessing import process
import torch
import torch.nn.functional as F
import math
import pandas as pd
import random
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

def compute_golden_loss(hidden_states_ls, golden_hidden_ls, hidden_mask, golden_mask,token_level_scores,mlp,align_type,loss_type,normalize,uid,config=None):
    h1 = hidden_states_ls  # (bsz, seq_len, hidden_size)
    h2 = golden_hidden_ls  # (bsz, seq_len, hidden_size)
    if config.get("add_mlp",False):
        h1=mlp(h1)
    if config.get("add_mlp_golden",False):
        with torch.no_grad():
            h2=mlp(h2)
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