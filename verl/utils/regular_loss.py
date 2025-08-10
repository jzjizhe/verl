from multiprocessing import process
import torch
import torch.nn.functional as F
import math
import pandas as pd
import random

def process_hidden(hidden_states_ls):
    return hidden_states_ls[0]

def compute_golden_loss(hidden_states_ls, golden_hidden_ls, hidden_mask, golden_mask,token_level_scores,mlp,align_type,loss_type,normalize,uid,config=None):
    h1 = process_hidden(hidden_states_ls)  # (bsz, seq_len, hidden_size)
    if config.golden_from!="ref":
        h2 = process_hidden(golden_hidden_ls)  # (bsz, seq_len, hidden_size)
    if align_type=="last_token":
        last_token_indices=hidden_mask.sum(dim=1)-1
        batch_indices = torch.arange(h1.size(0), device=h1.device)
        h1 = h1[batch_indices, last_token_indices] # (bsz, hidden_size)
        if config.golden_from!="ref":
            last_golden_indices=golden_mask.sum(dim=1)-1
            golden_batch_indices = torch.arange(h2.size(0), device=h2.device)
            h2 = h2[golden_batch_indices, last_golden_indices] # (bsz, hidden_size)
        else:
            h2=golden_hidden_ls
        if config.get("add_mlp",False):
            h1=mlp(h1)
    elif align_type=="token-2":
        last_token_indices=hidden_mask.sum(dim=1)-2
        batch_indices = torch.arange(h1.size(0), device=h1.device)
        h1 = h1[batch_indices, last_token_indices] # (bsz, hidden_size)
        if config.golden_from!="ref":
            last_golden_indices=golden_mask.sum(dim=1)-2
            golden_batch_indices = torch.arange(h2.size(0), device=h2.device)
            h2 = h2[golden_batch_indices, last_golden_indices] # (bsz, hidden_size)
        else:
            h2=golden_hidden_ls
        if config.get("add_mlp",False):
            h1=mlp(h1)
    elif align_type=="global_pooling":
        h1=h1.sum(dim=1)/hidden_mask.sum(dim=1,keepdim=True)
        h2=h2.sum(dim=1)/golden_mask.sum(dim=1,keepdim=True)
    elif align_type=="random_golden_bottom_k":
        # k=random.randint(0,9)
        k=torch.randint(0,10,size=(h1.size(0),),device=h1.device)
        last_token_indices=hidden_mask.sum(dim=1)-1
        batch_indices = torch.arange(h1.size(0), device=h1.device)
        h1 = h1[batch_indices, last_token_indices] # (bsz, hidden_size) 
        last_golden_indices=golden_mask.sum(dim=1)-1-k
        golden_batch_indices = torch.arange(h2.size(0), device=h2.device)
        h2 = h2[golden_batch_indices, last_golden_indices] # (bsz, hidden_size)
        if config.get("add_mlp",False):
            h1=mlp(h1)
    elif align_type=="all2last":
        last_golden_indices = golden_mask.sum(dim=1) - 1
        golden_batch_indices = torch.arange(h2.size(0), device=h2.device)
        h2 = h2[golden_batch_indices, last_golden_indices].unsqueeze(1)  # (bsz, 1, hidden_size)
    else:
        raise ValueError(f"Invalid alignment type: {align_type}")
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
        total_flip=flip_score.sum() 
        hidden_golden_loss = (hidden_golden_loss * flip_score).sum() / total_flip if total_flip > 0 else 0
    elif loss_type=="mse":
        hidden_golden_loss = F.mse_loss(h1, h2)
    elif loss_type=="cosine_wrong":
        cos_sim = F.cosine_similarity(h1, h2, dim=-1)
        flip_score = 1-token_level_scores.sum(-1)
        total_flip = flip_score.sum()
        hidden_golden_loss = 1 - (cos_sim * flip_score).sum() / total_flip if total_flip > 0 else 0
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