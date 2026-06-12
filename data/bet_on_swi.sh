#!/bin/bash
set -euo pipefail

SOURCE_DIR="/Volumes/MOVESPEED/QSM_multicohort_data/shunyi_followup_bids"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
  echo "错误: 源目录 '$SOURCE_DIR' 不存在"
  exit 1
fi

# 设置日志文件（保存在源目录下）
LOG_FILE="$SOURCE_DIR/bet2_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "开始处理 SWI 文件..."
echo "源目录: $SOURCE_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 使用 find 递归查找所有以 _SWI.nii.gz 结尾的文件
files=()
while IFS= read -r -d '' file; do
  files+=("$file")
done < <(find "$SOURCE_DIR" -type f -name "sub-*_SWI.nii.gz" -print0 2>/dev/null)

# 检查是否找到文件
if [ ${#files[@]} -eq 0 ]; then
  echo "未找到文件: $SOURCE_DIR 下没有找到 *_SWI.nii.gz 文件"
  exit 1
fi

echo "找到 ${#files[@]} 个 SWI 文件"
echo ""

# 处理每个文件
count=0
skipped=0
for f in "${files[@]}"; do
  if [ -f "$f" ]; then
    # 检查文件名是否以 '.' 开头
    filename=$(basename "$f")
    if [[ "$filename" == .* ]]; then
      skipped=$((skipped + 1))
      echo "跳过隐藏文件: $filename"
      continue
    fi
    
    count=$((count + 1))
    out="${f%.nii.gz}_brain"   # 输出: xxx_SWI_brain.nii.gz 和 xxx_SWI_brain_mask.nii.gz
    echo "[$count/${#files[@]}] 处理: $filename"
    echo "  输入: $f"
    echo "  输出: ${out}.nii.gz 和 ${out}_mask.nii.gz"
    bet2 "$f" "$out" -m
    echo "  ✓ 完成"
    echo ""
  else
    echo "警告: '$f' 不是有效文件，跳过"
  fi
done

echo "完成！共处理 $count 个文件，跳过 $skipped 个隐藏文件"
