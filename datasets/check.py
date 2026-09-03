"""
数据集诊断脚本：随机抽取若干张训练集图片，把 YOLO 标签框画上去，
生成一张拼图，用于肉眼检查标注是否正确。

用法：
    python check_dataset.py

只需要修改下面的 ROOT 为你 GTSRB_YOLO 的绝对路径即可。
"""
import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont

# ==================== 需要修改的地方 ====================
ROOT = "/home/monkey/yolo/YOLOv8-TrafficSign-CBAM/datasets/GTSRB/GTSRB_YOLO"
# ======================================================

LABEL_DIR = os.path.join(ROOT, "labels/train")
IMG_DIR = os.path.join(ROOT, "images/train")

# 类别名（与 data.yaml 对应）
NAMES = [
    "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60",
    "speed_limit_70", "speed_limit_80", "speed_limit_80_end", "speed_limit_100",
    "speed_limit_120", "no_passing", "no_passing_trucks", "right_of_way",
    "priority_road", "yield", "stop", "no_vehicles", "trucks_forbidden",
    "no_entry", "general_caution", "dangerous_curve_left", "dangerous_curve_right",
    "double_curve", "bumpy_road", "slippery_road", "road_narrows", "construction",
    "traffic_signals", "pedestrians", "children_crossing", "bicycles_crossing",
    "snow", "animals", "end_of_all_limits", "turn_right_ahead", "turn_left_ahead",
    "ahead_only", "go_straight_or_right", "go_straight_or_left", "keep_right",
    "keep_left", "roundabout", "end_of_no_passing", "end_of_no_passing_trucks",
]

# 抽取数量 & 输出
NUM_SAMPLES = 9       # 3x3 拼图
OUTPUT = "check_dataset_result.jpg"
IMG_SIZE = 416        # 拼图里每张图的显示尺寸


def get_color(cls_id):
    """根据类别给一个固定颜色"""
    random.seed(cls_id)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def draw_labels(img_path, label_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h

                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

                color = get_color(cls_id)
                # 画框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                # 写类别名
                name = NAMES[cls_id] if cls_id < len(NAMES) else str(cls_id)
                draw.text((x1, max(0, y1 - 12)), name, fill=color)
    else:
        # 没有对应标签文件
        draw.text((5, 5), "NO LABEL", fill=(255, 0, 0))

    return img


def main():
    # 找所有标签文件
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    print(f"在 {LABEL_DIR} 下共找到 {len(label_files)} 个标签文件")

    if len(label_files) == 0:
        print("❌ 没有找到任何标签文件！请检查 LABEL_DIR 路径是否正确。")
        return

    # 随机抽几张
    samples = random.sample(label_files, min(NUM_SAMPLES, len(label_files)))

    thumbs = []
    for lbl in samples:
        base = os.path.splitext(os.path.basename(lbl))[0]
        # 图片可能是 .jpg（转换脚本生成的）
        img_path = os.path.join(IMG_DIR, base + ".jpg")
        if not os.path.exists(img_path):
            # 兜底：找任意同名前缀的图
            candidates = glob.glob(os.path.join(IMG_DIR, base + ".*"))
            img_path = candidates[0] if candidates else None

        if img_path and os.path.exists(img_path):
            rendered = draw_labels(img_path, lbl)
            rendered.thumbnail((IMG_SIZE, IMG_SIZE))
            thumbs.append(rendered)
            print(f"  ✅ {base}.jpg 标签数: {sum(1 for _ in open(lbl))}")
        else:
            print(f"  ❌ 找不到对应图片: {base}")

    # 拼成 3x3 网格
    if not thumbs:
        print("❌ 没有成功渲染任何图片，请检查 IMG_DIR 路径。")
        return

    grid_cols = 3
    grid_rows = (len(thumbs) + grid_cols - 1) // grid_cols
    grid = Image.new("RGB", (grid_cols * IMG_SIZE, grid_rows * IMG_SIZE), (255, 255, 255))
    for i, thumb in enumerate(thumbs):
        r = i // grid_cols
        c = i % grid_cols
        grid.paste(thumb, (c * IMG_SIZE, r * IMG_SIZE))

    grid.save(OUTPUT)
    print(f"\n✅ 诊断图已保存至: {os.path.abspath(OUTPUT)}")
    print("👉 打开它，检查：框是否正好套住标志？类别名是否正确？")


if __name__ == "__main__":
    main()
