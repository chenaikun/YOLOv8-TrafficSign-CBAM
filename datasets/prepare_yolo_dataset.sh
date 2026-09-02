#!/bin/bash

#源目录
SRC_DIR="../4/ts/ts"

#输出目录
DST_DIR="../traffic"

mkdir -p "$DST_DIR/images/train"
mkdir -p "$DST_DIR/images/val"
mkdir -p "$DST_DIR/labels/train"
mkdir -p "$DST_DIR/labels/val"

#收集并打乱数据名字
mapfile -t FILE_BASENAMES < <(find "$SRC_DIR" -type f -name "*.jpg" | xargs -n1 basename | sed 's/\.jpg//' | shuf)

TOTAL=${#FILE_BASENAMES[@]}
TRAIN_COUNT=$((TOTAL*80/100))


echo "总文件数量：$TOTAL"
echo "训练数量：$TRAIN_COUNT"
echo "验证数量：$((TOTAL-TRAIN_COUNT))"


#循环复制文件
for i in "${!FILE_BASENAMES[@]}"; do
    BASE="${FILE_BASENAMES[$i]}"

    if [ "$i" -lt "$TRAIN_COUNT" ]; then
        SPLIT="train" 
    else
        SPLIT="val"
    fi


    #复制图片
    cp "$SRC_DIR/$BASE.jpg" "$DST_DIR/images/$SPLIT/"

    #复制标签
    cp "$SRC_DIR/$BASE.txt" "$DST_DIR/labels/$SPLIT/"

done

echo "✅ 数据集整理和划分完成！输出目录: $DST_DIR"



