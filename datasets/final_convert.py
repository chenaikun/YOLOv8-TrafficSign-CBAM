import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

BASE_DIR = "/home/monkey/yolo/YOLOv8-TrafficSign-CBAM/datasets/GTSRB"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Final_Training/Images")
OUTPUT_ROOT = os.path.join(BASE_DIR, "GTSRB_YOLO")

if os.path.exists(OUTPUT_ROOT):
    print(f"❌ 错误：{OUTPUT_ROOT} 已存在！请先手动删除它再运行。")
    print(f"   运行: rm -rf {OUTPUT_ROOT}")
    exit(1)

print("正在读取所有类别标注...")
dfs = []
for class_id in range(43):
    class_folder = f"{class_id:05d}"
    csv_path = os.path.join(TRAIN_IMG_DIR, class_folder, f"GT-{class_folder}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=';')
        df['Filename'] = df['Filename'].apply(
            lambda x: os.path.join(TRAIN_IMG_DIR, class_folder, x)
        )
        dfs.append(df)
        print(f"  ✅ 类别 {class_folder}: {len(df)} 条")
    else:
        print(f"  ❌ 未找到: {csv_path}")

train_df = pd.concat(dfs, ignore_index=True)
print(f"\n✅ 总标注数: {len(train_df)}")

train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['ClassId']
)
print(f"训练: {len(train_split)}, 验证: {len(val_split)}")

for split_name, df in [("train", train_split), ("val", val_split)]:
    img_dir = os.path.join(OUTPUT_ROOT, "images", split_name)
    lbl_dir = os.path.join(OUTPUT_ROOT, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    grouped = df.groupby('Filename')
    for img_path, group in grouped:
        # ← 关键修复：加类别前缀，防止文件名冲突
        class_folder = os.path.basename(os.path.dirname(img_path))
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        fname = f"{class_folder}_{base_name}.jpg"

        dst = os.path.join(img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img.save(dst, quality=95)

        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        with open(lbl_path, "w") as f:
            for _, row in group.iterrows():
                w, h = row['Width'], row['Height']
                cx = ((row['Roi.X1'] + row['Roi.X2']) / 2) / w
                cy = ((row['Roi.Y1'] + row['Roi.Y2']) / 2) / h
                bw = (row['Roi.X2'] - row['Roi.X1']) / w
                bh = (row['Roi.Y2'] - row['Roi.Y1']) / h
                f.write(f"{int(row['ClassId'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"  ✅ {split_name}: {len(grouped)} 张图")

print(f"\n🎉 完成！输出至: {OUTPUT_ROOT}")