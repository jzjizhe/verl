
from huggingface_hub import HfApi

api = HfApi()
# api.upload_file(
#     path_or_fileobj="/data0/jzzhang/verl/datasets/NuminaMath-CoT/NuminaMath171k_test_math_format.parquet",
#     path_in_repo="NuminaMath171k_test_math_format.parquet",  # 在仓库中的保存路径
#     repo_id="AstirPair/NuminaMath171k_test_math_format",  # 例如："username/math-datasets"
#     repo_type="dataset"  # 仓库类型，数据集用 "dataset"，模型用 "model"
# 
api.upload_file(
    path_or_fileobj="/data0/jzzhang/verl/datasets/NuminaMath-CoT/NuminaMath171k_math_format.parquet",
    path_in_repo="NuminaMath171k_math_format.parquet",  # 在仓库中的保存路径
    repo_id="AstirPair/NuminaMath171k_math_format",  # 例如："username/math-datasets"
    repo_type="dataset"  # 仓库类型，数据集用 "dataset"，模型用 "model"
)