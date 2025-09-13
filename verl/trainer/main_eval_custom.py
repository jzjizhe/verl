
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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import hydra
import pandas as pd
import numpy as np
import math
from verl.utils.reward_score.math import compute_score
from verl.utils.fs import copy_to_local




@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
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
    
    # 同时打印到控制台和保存到文件
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # 保存到txt文件
    output_file = config.data.get("output_file", "evaluation_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
