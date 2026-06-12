#!/bin/bash
# Dataset should be format as 
# Center ID + Center Name
#   Subject ID + Subject Name
#           ...
#           Dicoms

# 获取输入参数
# SOURCE_DIR="/PATH/TO/DICOMS/DATASET"
# TARGET_DIR="/PATH/TO/SAVE"
# SOURCE_DIR="/Volumes/QSM/csvd-dicom"
# TARGET_DIR="/Volumes/QSM/csvd-bids"
SOURCE_DIR="/Volumes/CSVD/shunyi_followup_dcmmixed"
TARGET_DIR="/Volumes/CSVD/shunyi_followup_bids"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录 '$SOURCE_DIR' 不存在"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 设置日志文件
LOG_FILE="$TARGET_DIR/dcm2niix_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${LOG_FILE}")
exec 2> >(tee -a "${LOG_FILE}" >&2)

echo "日志将记录在: $LOG_FILE"

# 遍历源目录中的所有子目录

for dir in "$SOURCE_DIR"/*; do
    echo "处理目录: $dir"
    base_name=$(basename "$dir")
    # hpid=$(echo "$base_name" | cut -d'-' -f2)
    # pid=$(echo "$base_name" | cut -d'-' -f3)
    echo "处理患者目录: $subdir"
    if [ -d "$dir" ]; then
        # target_dir="$TARGET_DIR/$base_name"
        # mkdir -p "$target_dir"
        # echo "创建目标目录: $target_dir"
        # echo "处理目录: $dir"
        # dcm2niix -f "%s+%p+%t" -i y -d 9 -p n -z y -ba n -o "$target_dir" "$dir" 
        # dcm2bids -p "$hpid$pid" -d "$dir" -o "$target_dir" -c "bids_config.json"
        dcm2bids -p "$base_name" -d "$dir" -o "$TARGET_DIR" -c "bids_config.json"
    fi
done

# dcm2bids -p "$hpid$pid" -d "$dir" -o "$target_dir" -c "bids_config.json"

echo "转换完成！"