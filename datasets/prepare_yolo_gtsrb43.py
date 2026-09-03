# convert_gtsrb_v2.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

BASE_DIR = "/home/monkey/yolo/YOLOv8-TrafficSign-CBAM/datasets/GTSRB"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Final_Training/Images")
OUTPUT_ROOT = os.path.join(BASE_DIR, "GTSRB_YOLO")

print("正在读取训练集标注...")

dfs = []
for class_id in range(43):
    class_folder = format(class_id, '05d')
    csv_path = os.path.join(TRAIN_IMG_DIR, class_folder, f"GT-{class_folder}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=';')
        df['Filename'] = df['Filename'].apply(
            lambda x: os.path.join(TRAIN_IMG_DIR, class_folder, x)
        )
        dfs.append(df)
        print(f"  ✅ 类别 {class_folder}: {len(df)} 条")  # ← 改进：打印进度
    else:
        print(f"  ❌ 未找到: {csv_path}")

train_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
print(f"\n训练集总图片数：{len(train_df)}")  # 应该打印 39000+

# 划分
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['ClassId']
)
print(f"划分完成 → 训练: {len(train_split)}, 验证: {len(val_split)}")

def convert_and_copy(df, split_name):
    img_out_dir = os.path.join(OUTPUT_ROOT, "images", split_name)
    label_out_dir = os.path.join(OUTPUT_ROOT, "labels", split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    # 按图片分组，避免追加模式重复写
    grouped = df.groupby('Filename')
    
    for i, (img_path, group) in enumerate(grouped):
        if not os.path.exists(img_path):
            continue
        
        fname = os.path.splitext(os.path.basename(img_path))[0] + ".jpg"
        dst_img = os.path.join(img_out_dir, fname)
        
        # 转 jpg
        if not os.path.exists(dst_img):
            img = Image.open(img_path).convert("RGB")
            img.save(dst_img, quality=95)
        
        # 写标签（用 "w" 覆盖写，每组只写一次）
        label_path = os.path.join(label_out_dir, os.path.splitext(fname)[0] + ".txt")
        with open(label_path, "w") as f:  # ← 改进：用 "w" 而不是 "a"
            for _, row in group.iterrows():
                w, h = row['Width'], row['Height']
                cx = ((row['Roi.X1'] + row['Roi.X2']) / 2) / w
                cy = ((row['Roi.Y1'] + row['Roi.Y2']) / 2) / h
                bw = (row['Roi.X2'] - row['Roi.X1']) / w
                bh = (row['Roi.Y2'] - row['Roi.Y1']) / h
                f.write(f"{int(row['ClassId'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    
    print(f"  ✅ {split_name} 完成 ({len(grouped)} 张图)")

print("\n处理训练集...")
convert_and_copy(train_split, "train")

print("处理验证集...")
convert_and_copy(val_split, "val")

print(f"\n🎉 全部完成！输出至: {OUTPUT_ROOT}")