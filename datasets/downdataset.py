import kagglehub
import os

#选择下载路径
download_path=os.getcwd()+"/datasets"
#print(download_path)

#确保目录存在
os.makedirs(download_path,exist_ok=True)

#下载数据
path=kagglehub.dataset_download("valentynsichkar/traffic-signs-dataset-in-yolo-format")

#移动到目标路径
os.rename(path,os.path.join(download_path,os.path.basename(path)))

print("下载完成！路径：",os.path.join(download_path,os.path.basename(path)))

