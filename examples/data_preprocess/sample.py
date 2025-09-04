from datasets import load_dataset

# 加载指定路径下的数据集
dataset = load_dataset("parquet", data_dir="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data")

# 随机采样171000条数据
if "train" in dataset:
    train_dataset = dataset["train"]
# else:
    # 如果没有train split，取第一个split
    # train_dataset = dataset[list(dataset.keys())[0]]

# sampled_dataset = train_dataset.shuffle(seed=42).select(range(171000))
sampled_dataset = train_dataset.shuffle(seed=42)


# 保存为新的parquet文件
output_path = "/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/train_shuffle.parquet"
sampled_dataset.to_parquet(output_path)

print(f"save path {output_path}")
# import pdb;pdb.set_trace()

