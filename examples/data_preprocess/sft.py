import pandas as pd
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))
file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet"

data_list = pd.read_parquet(file_path).to_dict(orient="records")
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
math_format_data=[]
total=0
for idx,item in enumerate(data_list):
    try:
        new_item={}
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        # import pdb;pdb.set_trace()
        question_raw=item["problem"]
        answer_raw=item['solution']
        solution = extract_solution(answer_raw)

        question = item["problem"] + " " + instruction_following
        # print(idx)
        new_item = {
            "data_source": 'math_verify_numina',
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "train", "index": total,
            "answer":answer_raw,
            'question':question_raw}
        }
        math_format_data.append(new_item)
        total+=1
        if total==171000:
            break
    except:
        print(idx)
        # import pdb;pdb.set_trace()
df = pd.DataFrame(math_format_data)
df.to_parquet("/data0/jzzhang/verl/datasets/NuminaMath-CoT/NuminaMath171k_SFT_test_math_format.parquet", index=False)
    
    
    