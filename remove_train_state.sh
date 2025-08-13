#!/bin/bash

# 递归删除指定目录下所有包含"optim_world_size"的文件
# 使用方法: ./delete_optim_files.sh [目录路径]

# 检查是否提供了目录参数，默认为当前目录
TARGET_DIR="${1:-/data1/hhzhang/improve}"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 '$TARGET_DIR' 不存在。" >&2
    exit 1
fi

# 递归查找匹配的文件（find命令默认递归搜索所有子目录）
echo "正在递归搜索目录: $TARGET_DIR"
FILES_TO_DELETE=$(find "$TARGET_DIR" -type f -name "*optim_world_size*" 2>/dev/null | sort)

# 检查是否有匹配的文件
if [ -z "$FILES_TO_DELETE" ]; then
    echo "没有找到包含 'optim_world_size' 的文件。"
    exit 0
fi

# 显示待删除的文件
echo "找到以下文件:"
echo "$FILES_TO_DELETE" | nl
echo "总计: $(echo "$FILES_TO_DELETE" | wc -l) 个文件"

# 确认删除
read -p "确定要删除这些文件吗？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消。"
    exit 0
fi

# 执行删除（使用find结合-exec）
echo "$FILES_TO_DELETE" | while IFS= read -r file; do
    echo "删除: $file"
    rm -f "$file" || echo "警告: 无法删除 $file" >&2
done

echo "操作完成。"