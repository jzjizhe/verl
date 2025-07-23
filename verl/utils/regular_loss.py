from multiprocessing import process
import torch
import torch.nn.functional as F

def process_hidden(hidden_states_ls):
    return hidden_states_ls[0]

def compute_golden_loss(hidden_states_ls, golden_hidden_ls, hidden_mask, golden_mask):
    # 只对第一个层做正则
    # import pdb;pdb.set_trace()
    hidden_length = hidden_states_ls[0].size(1)
    golden_length = golden_hidden_ls[0].size(1)
    hidden_mask=hidden_mask[:,-hidden_length:]
    golden_mask=golden_mask[:,-golden_length:]
    h1 = process_hidden(hidden_states_ls)  # (bsz, seq_len, hidden_size)
    h2 = process_hidden(golden_hidden_ls)  # (bsz, seq_len, hidden_size)
    # mask: (bsz, seq_len)
    valid_mask = hidden_mask.bool() & golden_mask.bool()  # 只对都有效的位置
    h1_valid = h1[valid_mask]
    h2_valid = h2[valid_mask]
    cos_sim = F.cosine_similarity(h1_valid, h2_valid, dim=-1)
    hidden_golden_loss = 1 - cos_sim.mean()
    return hidden_golden_loss
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