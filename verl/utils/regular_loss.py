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
    h1 = hidden_states_ls  # (bsz, layers, seq_len, hidden_dim)
    h2 = golden_hidden_ls  # (bsz, layers, seq_len, hidden_dim)
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
    elif loss_type=="attention":
        hidden_golden_loss=cross_attention_loss(h1[:,0,:,:],h2[:,0,:,:],hidden_mask,golden_mask,temperature=0.1)
        print(hidden_golden_loss)
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
    支持padding和mask的交叉对齐损失函数
    
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
    
    # 处理mask，默认全为1（有效）
    # if gen_mask is None:
    #     gen_mask = torch.ones((batch_size, gen_seq_len), dtype=torch.float32, device=gen_hidden.device)
    # if gold_mask is None:
    #     gold_mask = torch.ones((batch_size, gold_seq_len), dtype=torch.float32, device=gold_hidden.device)
    
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
    
    # 计算生成序列hidden state与对齐向量的MSE损失，忽略padding部分
    # 将gen_mask扩展为[batch, gen_seq_len, 1]用于广播
    gen_mask_expanded = gen_mask.unsqueeze(-1)  # [batch, gen_seq_len, 1]
    
    # 只计算有效token的损失
    mse = F.mse_loss(gen_hidden, aligned_gold_hidden, reduction='none')  # [batch, gen_seq_len, dim]
    mse = mse * gen_mask_expanded  # 屏蔽padding部分的损失
    
    # 计算平均损失，除以有效token的数量
    valid_count = gen_mask.sum() * dim  # 总有效元素数量
    loss = mse.sum() / valid_count.clamp(min=1e-8)  # 避免除以0
    
    return loss