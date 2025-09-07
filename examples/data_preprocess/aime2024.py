
import pandas as pd

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))
# 读取 parquet 文件
# file_path = "/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/train_sampled_171k.parquet"
file_path="/data0/jzzhang/verl/datasets/aime24/aime_2024_problems.parquet"
data_list = pd.read_parquet(file_path).to_dict(orient="records")
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
math_format_data=[]
total=0
for idx,item in enumerate(data_list):
    try:
        new_item={}
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        question = item["Problem"] + " " + instruction_following
        # print(item['solution'])
        print(idx)
        solution = str(item['Answer'])
        new_item = {
            "data_source": 'math_verify_aime2024',
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "golden_answer":item['Solution'],
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": total},
        }
        math_format_data.append(new_item)
        total+=1
    except:
        print(idx)
        # import pdb;pdb.set_trace()
df = pd.DataFrame(math_format_data*3)
print(len(math_format_data*3))
df.to_parquet("/data0/jzzhang/verl/datasets/aime24/aime2024_math_format_repeat3.parquet", index=False)
    
    
    