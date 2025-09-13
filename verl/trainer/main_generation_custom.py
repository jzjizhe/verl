# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import math
import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.reward_score.math import compute_score

@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))

def evaluation(dataset,config):
    responses = dataset[config.data.response_key]
    reward_model_data = dataset[config.data.reward_model_key]
    total = len(dataset)
    # import pdb;pdb.set_trace()

    n_responses = len(responses[0])  # 假设每个prompt的response数量相同
    scores = []
    for i in range(total):
        gt = reward_model_data[i]['ground_truth']
        resp_scores = []
        for j in range(n_responses):
            resp_scores.append(compute_score(responses[i][j], gt))
        scores.append(resp_scores)

    # 计算 mean@32, 方差
    means = [np.mean(s) for s in scores]
    vars_ = [np.var(s) for s in scores]
    mean_at_32 = np.mean(means)
    var_at_32 = np.mean(vars_)

    # 准备输出信息
    output_lines = []
    output_lines.append(f"mean@{n_responses}: {mean_at_32}")
    output_lines.append(f"var@{n_responses}: {var_at_32}")

    # 计算 pass@2^n
    def pass_at_k(num_correct, num_total, k):
        if num_total < k:
            return 1.0 if num_correct > 0 else 0.0
        res = 1.0
        for i in range(k):
            res *= (num_total - num_correct - i) / (num_total - i)
        return 1 - res

    max_pow = int(math.log2(n_responses))
    for exp in range(0, max_pow + 1):
        k = 2 ** exp
        pass_count = 0
        for i in range(total):
            correct = sum(scores[i])
            pass_count += pass_at_k(correct, n_responses, k)
        pass_at_k_mean = pass_count / total
        output_lines.append(f"pass@{k}: {pass_at_k_mean}")
    return output_lines
    # 同时打印到控制台和保存到文件
    # output_text = "\n".join(output_lines)
    # print(output_text)
    
    # 保存到txt文件
    # output_file = config.data.get("output_file", "evaluation_results.txt")
    # with open(output_file, "w", encoding="utf-8") as f:
        # f.write(output_text)
    # print(f"结果已保存到: {output_file}")



@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    datasets_path=config.data.path
    output_info=[]
    for path in datasets_path:
        file_name=os.path.basename(path)
        process_info=f"==============={path} is processing==============="

        output_info.append(process_info)
        print(process_info)
        
        dataset = pd.read_parquet(path)
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [chat.tolist() for chat in chat_lst]
        total_samples = len(dataset)
        config_batch_size = config.data.batch_size
        num_batch = -(-total_samples // config_batch_size)
        output_lst = [[] for _ in range(config.data.n_samples)]

        for batch_idx in range(num_batch):
            print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
            batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
            inputs = tokenizer.apply_chat_template(
                batch_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

            data = DataProto.from_dict(batch_dict)
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

            # START TO GENERATE FOR n_samples TIMES
            print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
            for n_sample in range(config.data.n_samples):
                output_padded = wg.generate_sequences(data_padded)
                output = unpad_dataproto(output_padded, pad_size=pad_size)

                output_texts = []
                for i in range(len(output)):
                    data_item = output[i]
                    prompt_length = data_item.batch["prompts"].shape[-1]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = data_item.batch["responses"][:valid_response_length]
                    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    output_texts.append(response_str)

                output_lst[n_sample].extend(output_texts)

        # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
        output_lst = np.array(output_lst, dtype=object)
        output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

        # add to the data frame
        dataset["responses"] = output_lst
        # output_dir = os.path.dirname(config.data.output_path)
        makedirs(config.data.output_path, exist_ok=True)
        outpath=os.path.join(config.data.output_path,file_name)
        dataset.to_parquet(outpath)
        output_info+=evaluation(dataset,config)
    # 同时打印到控制台和保存到文件
    output_text = "\n".join(output_info)
    # print(output_text)
    # 保存到txt文件
    output_file = config.data.get("output_file", "evaluation_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
