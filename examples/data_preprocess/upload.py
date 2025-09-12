
from huggingface_hub import HfApi

api = HfApi()
# api.upload_file(
#     path_or_fileobj="/data0/jzzhang/verl/datasets/numinamath-cot/numinamath171k_test_math_format.parquet",
#     path_in_repo="numinamath171k_test_math_format.parquet",  # 在仓库中的保存路径
#     repo_id="astirpair/numinamath171k_test_math_format",  # 例如："username/math-datasets"
#     repo_type="dataset"  # 仓库类型，数据集用 "dataset"，模型用 "model"
# 
# api.upload_file(
#     path_or_fileobj="/data0/jzzhang/verl/datasets/aime24/aime2024_math_format_repeat3.parquet",
#     path_in_repo="aime2024_math_format_repeat3.parquet",  # 在仓库中的保存路径
#     repo_id="AstirPair/aime2024_3",  # 例如："username/math-datasets"
#     repo_type="dataset"  # 仓库类型，数据集用 "dataset"，模型用 "model"
# )
# api.upload_file(
#     path_or_fileobj="/data0/jzzhang/verl/datasets/datasets",
#     path_in_repo="RARL",  # 在仓库中的保存路径
#     repo_id="AstirPair/RARL",  # 例如："username/math-datasets"
#     repo_type="dataset"  # 仓库类型，数据集用 "dataset"，模型用 "model"
# )

from huggingface_hub import HfApi

api = HfApi()

# 上传整个文件夹
api.upload_folder(
    folder_path="/data0/jzzhang/verl/datasets/datasets",                # 本地文件夹
    repo_id="AstirPair/RARL",          # 仓库
    repo_type="dataset"                       # 类型：model/dataset/space
)
