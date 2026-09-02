#用于将原始GTSRB数据集转换成yolov8专用格式

import os
#文件操作工具
import shutil
#处理csv
import pandas as pd
#将数据分成验证集和验证集
from sklearn.model_selection import train_test_split
#导入PIL图片库，用于把ppm转换成jpg
from PIL import Image

#设置文件路径
BASE_DIR="/home/monkey/yolo/YOLOv8-TrafficSign-CBAM/datasets/GTSRB"
#训练图片路径
TRAIN_IMG_DIR=os.path.join(BASE_DIR,"Final_Training/Images")
#测试图片路径
TEST_IMG_DIR=os.path.join(BASE_DIR,"Final_Test/Images")
#测试集的标注文件
TEST_GT_PATH=os.path.join(BASE_DIR,"Final_Test/GT-final_test.csv")

#输出转换后的yolov8输出目录
OUTPUT_ROOT=os.path.join(BASE_DIR,"GTSRB_YOLO")

#读取并合并训练集标注

print("正在读取训练集标注...")


dfs=[]

for class_id in range(43):
    class_folder=format(class_id,'05d')
    csv_path=os.path.join(TRAIN_IMG_DIR,class_folder,f"GT-{class_folder}.csv")

    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path,sep=';')
        #print(df)
        df['Filename']=df['Filename'].apply(lambda x:os.path.join(TRAIN_IMG_DIR,class_folder,x))

        dfs.append(df)
    else:
        print(f"警告：未找到{csv_path}")


train_df=pd.concat(dfs,ignore_index=True) if dfs else pd.DataFrame()

print(f"训练集总图片数：{len(train_df)}")

#划分训练集和验证集
print("划分训练集和验证集...")

train_split,val_split=train_test_split(train_df,test_size=0.2,random_state=42,stratify=train_df['ClassId'])

#定义转换函数
def convert_and_copy(df,split_name):
    img_out_dir=os.path.join(OUTPUT_ROOT,"images",split_name)
    label_out_dir=os.path.join(OUTPUT_ROOT,"labels",split_name)

    os.makedirs(img_out_dir,exist_ok=True)
    os.makedirs(label_out_dir,exist_ok=True)

    for _, row in df.iterrows():
        img_path=row['Filename']

        if not os.path.exists(img_path):
            continue
        fname=os.path.splitext(os.path.basename(img_path))[0]+".jpg"
        dst_img=os.path.join(img_out_dir,fname)

        if not os.path.exists(dst_img):
            img=Image.open(img_path)
            img.save(dst_img)

        #计算yolo格式标签
        w,h=row['Width'],row['Height']
        cx=((row['Roi.X1'] + row['Roi.X2']) / 2) / w
        cy=((row['Roi.Y1'] + row['Roi.Y2']) / 2) / h
        bw = (row['Roi.X2'] - row['Roi.X1']) / w
        bh = (row['Roi.Y2'] - row['Roi.Y1']) / h

        class_id=row['ClassId']
        label_path = os.path.join(label_out_dir, os.path.splitext(fname)[0] + ".txt")
        with open(label_path,"a") as f:
            # 按 YOLO 要求的格式写入一行：类别ID 中心点X 中心点Y 框宽 框高
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


#执行

print("处理训练集...")

convert_and_copy(train_split,"train")

print("处理验证集...")
convert_and_copy(val_split,"val")



print("转换完成！数据已保存至:",OUTPUT_ROOT)