# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.utils.regular_loss import compute_golden_loss
from verl.utils.custom_print import rank_zero_print
from verl.workers.actor import BasePPOActor
import math
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
def find_special_positions(tensor, token_id, mask):
    """
    Args:
        tensor: 2D tensor [batch_size, seq_len]
        token_id: 目标token的ID
        mask: padding掩码 (1表示有效，0表示padding)
    
    Returns:
        1D tensor [batch_size] 包含每行最后一个token_id位置，
        若不存在则返回pad_prev_pos，若差值超过10则强制返回pad_prev_pos
    """
    seq_len = tensor.size(1)
    
    # 计算最后一个token_id位置
    mask_token = (tensor == token_id)
    flipped_token = mask_token.flip(dims=[1])
    last_token_idx_rev = flipped_token.int().argmax(dim=1)
    last_token_pos = seq_len - 1 - last_token_idx_rev
    
    # 计算有效长度末尾位置
    pad_prev_pos = mask.sum(dim=1) - 1
    
    # 初步结果：存在token_id则用其位置，否则用pad_prev_pos
    has_token = mask_token.any(dim=1)
    result = torch.where(has_token, last_token_pos, pad_prev_pos)
    
    # 强制覆盖条件：当差值超过10时使用pad_prev_pos
    delta = pad_prev_pos - last_token_pos
    override_mask = (delta > 20)
    result = torch.where(override_mask, pad_prev_pos, result)
    
    return result

# def find_special_positions(tensor, token_id,mask):
#     """ 返回每行最后token_id位置，若无则返回首个pad_id前位置 """
#     # 处理最后出现的token_id
#     mask_token = (tensor == token_id)
#     has_token = mask_token.any(dim=1)
    
#     # 反转后获取最后一个token_id位置
#     flipped_token = mask_token.flip(dims=[1])
#     last_token_idx_rev = flipped_token.int().argmax(dim=1)  # 反转维度后第一个True位置
#     last_token_pos = tensor.size(1) - 1 - last_token_idx_rev

#     pad_prev_pos=mask.sum(dim=1)-1
#     result = torch.where(has_token, last_token_pos, pad_prev_pos)
#     return result
def get_token_hidden_states(hidden_states_ls,align_type,mask,input_ids,token_id=79075):
    if align_type=="last_token":
        token_indices=mask.sum(dim=1)-1
    elif align_type=="token-2":
        token_indices=mask.sum(dim=1)-2
    elif align_type=="box_token":
        token_indices=find_special_positions(input_ids,token_id,mask)
    else:
        raise ValueError(f"Invalid alignment type: {align_type}")
    batch_indices = torch.arange(hidden_states_ls[0].size(0), device=hidden_states_ls[0].device)
    hidden_states_ls = [hidden_states[batch_indices, token_indices] for hidden_states in hidden_states_ls]
    hidden_states_ls = torch.stack(hidden_states_ls,dim=0).transpose(0,1)
    return hidden_states_ls

class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.get_hidden_state = self.config.get("get_hidden_state", False)
        # InfoNCE相关参数
        self.use_infonce_loss = self.config.get("use_infonce_loss", False)
        self.infonce_weight = self.config.get("infonce_weight", 0.1)
        self.infonce_temperature = self.config.get("infonce_temperature", 0.1)
        # golden hidden regular
        self.use_golden_loss=self.config.get("use_golden_loss",False)
        self.golden_loss_weight=self.config.get("golden_loss_weight",0.001)
        self.layer_list = self.config.get("layer_list", None)  # 新增layer_list参数
        # add mlp
        self.add_mlp=self.config.get("add_mlp",False)
        self.device_name = get_device_name()
        if self.use_golden_loss:
            rank_zero_print("use golden loss")
        if self.use_infonce_loss:
            rank_zero_print("use infonce loss")
        if self.add_mlp:
            rank_zero_print("add mlp")
        
    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, output_hidden_states=False,token_idx=None, layer_list=None):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            hidden_states: # (bs, response_len, hidden_size) or None, 如果layer_list不为空，返回layer_list[0]对应的层
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            hidden_states_ls = []
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                # input_ids.size()=[4,2048]
                # input_ids.unsqueeze(-1).size()=[4,2048,1]
                # attention_mask.size=[4,2048]
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                # input_ids_rmpad.size()

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                # Add output_hidden_states parameter
                if output_hidden_states:
                    extra_args["output_hidden_states"] = True
                    if layer_list is not None:
                        extra_args["output_hidden_states_layers"] = layer_list

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                # output.logits.size() [1,2934,151936]
                # output.hidden_states[0]: [1,2934,896]
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # Extract hidden states if requested
                if output_hidden_states and hasattr(output, 'hidden_states'):
                    if layer_list is not None and len(layer_list) > 0:
                        # 如果指定了layer_list，使用layer_list[0]对应的层
                        target_layer = layer_list
                        for target in layer_list:
                            hidden_states_ls.append(output.hidden_states[target].squeeze(0))

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if output_hidden_states:
                        for i in range(len(hidden_states_ls)):
                            hidden_states_ls[i] = gather_outputs_and_unpad(
                                hidden_states_ls[i],
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if output_hidden_states:
                    for i in range(len(hidden_states_ls)):
                        hidden_states_ls[i] = pad_input(
                            hidden_states=hidden_states_ls[i],
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if output_hidden_states:
                    hidden_states_ls=[item[:,-response_length:,:] for item in hidden_states_ls]
                    hidden_states_ls=get_token_hidden_states(hidden_states_ls,
                    align_type=self.config.align_type,
                    mask=micro_batch["response_mask"],
                    input_ids=input_ids[:,-response_length:])


            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                
                # Add output_hidden_states parameter
                if output_hidden_states:
                    extra_args["output_hidden_states"] = True
                    if layer_list is not None:
                        extra_args["output_hidden_states_layers"] = layer_list
                    
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                # Extract hidden states if requested
                if output_hidden_states and hasattr(output, 'hidden_states'):
                    if layer_list is not None and len(layer_list) > 0:
                        # 如果指定了layer_list，使用layer_list[0]对应的层
                        # target_layer = layer_list[0]
                        # if isinstance(output.hidden_states, (list, tuple)):
                            # 如果hidden_states是列表，直接索引
                        for target_layer in layer_list:
                            hidden_states_ls.append(output.hidden_states[target_layer][:, -response_length:, :])  # (bsz, response_length, hidden_size)

            return entropy, log_probs, hidden_states_ls

    def _forward_micro_batch_golden_response(self, micro_batch, temperature, calculate_entropy=False, output_hidden_states=False, layer_list=None):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            hidden_states: # (bs, response_len, hidden_size) or None, 如果layer_list不为空，返回layer_list[0]对应的层
        """
        response_length = micro_batch["golden_answer_ids"].size(-1)
        # mirco_batch["input_ids"]中包含prompt和model rollout的结果，需要将prompt单独拿出来与 golden response合并
        prompt_length = micro_batch["input_ids"].size(1)-micro_batch['responses'].size(1)
        input_ids = micro_batch["input_ids"][:,0:prompt_length]
        attention_mask = micro_batch["attention_mask"][:,0:prompt_length]
        position_ids = micro_batch["position_ids"][:,0:prompt_length]
        response=micro_batch['golden_answer_ids']
        input_ids = torch.cat([input_ids, response], dim=-1)
        batch_size, seqlen = input_ids.shape
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        # response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        response_attention_mask = micro_batch['golden_answer_attention_mask']
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # input_ids = micro_batch["golden_answer_ids"]
            # attention_mask = micro_batch["golden_answer_attention_mask"]
            # position_ids = micro_batch["golden_answer_position_ids"]
            entropy = None
            hidden_states_ls = []
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                # input_ids.size()=[4,2048]
                # input_ids.unsqueeze(-1).size()=[4,2048,1]
                # attention_mask.size=[4,2048]
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                # input_ids_rmpad.size()

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                # Add output_hidden_states parameter
                if output_hidden_states:
                    extra_args["output_hidden_states"] = True
                    if layer_list is not None:
                        extra_args["output_hidden_states_layers"] = layer_list

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                # output.logits.size() [1,2934,151936]
                # output.hidden_states[0]: [1,2934,896]
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # Extract hidden states if requested
                if output_hidden_states and hasattr(output, 'hidden_states'):
                    if layer_list is not None and len(layer_list) > 0:
                        target_layer = layer_list
                        for target in layer_list:
                            hidden_states_ls.append(output.hidden_states[target].squeeze(0))

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if output_hidden_states:
                        for i in range(len(hidden_states_ls)):
                            hidden_states_ls[i] = gather_outputs_and_unpad(
                                hidden_states_ls[i],
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if output_hidden_states:
                    for i in range(len(hidden_states_ls)):
                        hidden_states_ls[i] = pad_input(
                            hidden_states=hidden_states_ls[i],
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if output_hidden_states:
                    # 获取的是response 的hiden state,而不是预测next token的状态
                    hidden_states_ls=[item[:,-response_length:,:] for item in hidden_states_ls]
                    hidden_states_ls=get_token_hidden_states(hidden_states_ls,
                    align_type=self.config.align_type,
                    mask=micro_batch["golden_answer_attention_mask"],
                    input_ids=micro_batch["golden_answer_ids"])
                    # hidden_states_ls=[item[:,-response_length-1:-1,:] for item in hidden_states_ls]
            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                
                # Add output_hidden_states parameter
                if output_hidden_states:
                    extra_args["output_hidden_states"] = True
                    if layer_list is not None:
                        extra_args["output_hidden_states_layers"] = layer_list
                    
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                # Extract hidden states if requested
                if output_hidden_states and hasattr(output, 'hidden_states'):
                    if layer_list is not None and len(layer_list) > 0:
                        for target_layer in layer_list:
                            hidden_states_ls.append(output.hidden_states[target_layer][:, -response_length:, :])  # (bsz, response_length, hidden_size)
            return entropy, log_probs, hidden_states_ls

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs,_ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_golden_hidden_states(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys+=['golden_answer_ids','golden_answer_position_ids','golden_answer_attention_mask']
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        # log_probs_lst = []
        # entropy_lst = []
        hidden_states_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs,hidden_states_ls = self._forward_micro_batch_golden_response(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                    output_hidden_states=True,
                    layer_list=self.layer_list
                )
            # log_probs_lst.append(log_probs)
            # if calculate_entropy:
                # entropy_lst.append(entropy)
            h=hidden_states_ls
            # last_golden_indices=model_inputs["golden_answer_attention_mask"].sum(dim=1)-1
            # golden_batch_indices = torch.arange(h.size(0), device=h.device)
            # h = h[golden_batch_indices, last_golden_indices] # (bsz, hidden_size)
            hidden_states_lst.append(h)
        # log_probs = torch.concat(log_probs_lst, dim=0)
        hidden_states = torch.concat(hidden_states_lst, dim=0)
        # entropys = None
        # if calculate_entropy:
            # entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            hidden_states = restore_dynamic_batch(hidden_states, batch_idx_list)
            # log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            # if calculate_entropy:
                # entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return hidden_states

    def golden_loss_weight_schedule(self,step,loss_weight,total_steps,scheduler_type="cosine"):
        step-=1 # 0-indexed
        if scheduler_type=="cosine":
            eta_min = 0
            return eta_min + (loss_weight - eta_min) * (1 + math.cos(math.pi * step / total_steps)) / 2
        elif scheduler_type=="linear":
            return loss_weight * (1 - step / total_steps)
        elif scheduler_type=="constant":
            return loss_weight
        elif scheduler_type=="stage":
            if step < 10:
                return 0.005
            elif step < 30:
                return 0.001
            else:
                return 0
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "token_level_scores",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.use_golden_loss:
            step = data.meta_info["step"]
            total_steps = data.meta_info["total_steps"]
            select_keys+=['golden_answer_ids','golden_answer_position_ids','golden_answer_attention_mask']
            golden_loss_weight = self.golden_loss_weight_schedule(step,self.golden_loss_weight,total_steps,scheduler_type=self.config.get("golden_loss_scheduler_type","constant"))
        if self.config.golden_from=="ref":
            select_keys.append("golden_ref_hidden_states")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        non_tensor_select_keys.append("uid")
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, hidden_states_ls = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                        output_hidden_states=self.get_hidden_state,
                        layer_list=self.layer_list
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss
                        
                    if self.use_golden_loss:
                        if self.config.golden_from=="ref":
                            golden_hidden_ls=model_inputs["golden_ref_hidden_states"]
                        else:
                            with torch.no_grad():
                                golden_entropy,golden_log_prob,golden_hidden_ls= self._forward_micro_batch_golden_response(
                                    model_inputs,
                                    temperature=temperature,
                                    calculate_entropy=calculate_entropy,
                                    output_hidden_states=self.get_hidden_state,
                                    layer_list=self.layer_list
                                )
                        # if self.add_mlp:
                        #     hidden_states_ls[0]=self.actor_module.custom_mlp(hidden_states_ls[0])
                        # if self.config.get("mlp_golden",False):
                        #     with torch.no_grad():
                        #         golden_hidden_ls[0]=self.actor_module.custom_mlp(golden_hidden_ls[0])
                        # === hidden states cosine similarity regularization ===
                        hidden_golden_loss = compute_golden_loss(
                            hidden_states_ls,
                            golden_hidden_ls,
                            response_mask,
                            model_inputs["golden_answer_attention_mask"],
                            model_inputs["token_level_scores"],
                            mlp=self.actor_module.custom_mlp if self.add_mlp else None,
                            align_type=self.config.get("align_type","last_token"),
                            loss_type=self.config.get("loss_type","cosine"),
                            normalize=self.config.get("norm_embeddings",False),
                            uid=model_inputs["uid"],
                            config=self.config
                        )
                        del golden_hidden_ls
                        del hidden_states_ls
                        policy_loss = policy_loss + hidden_golden_loss * golden_loss_weight
                        metrics["actor/hidden_golden_loss"] = hidden_golden_loss.detach().item()
                        metrics["actor/hidden_golden_weight"] = golden_loss_weight
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/policy_loss": policy_loss.detach().item(),
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
