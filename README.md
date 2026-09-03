# YOLOv8-TrafficSign-CBAM
基于YOLOv8的实时交通标志检测

## 🛠️ 环境配置

- OS: Ubuntu 26.04LTS
- Python: 3.10
- Conda 环境: `yolo`
- 核心依赖:
  - torch
  - ultralytics
  - opencv-python
  - pandas
  - scikit-learn

## 📊 数据集

- **名称**: GTSRB (German Traffic Sign Recognition Benchmark)
- **类别数**: 43
- **训练集**: 约 39,209 张（8:2 划分后训练集约 31,367 张）
- **验证集**: 约 7,842 张
- **数据格式**: YOLO（转换脚本 `prepare_yolo_gtsrb43.py`）

## 🧪 实验记录

### Exp-01: YOLOv8n Baseline（无改进模块）

| 项目 | 设置 |
|------|------|
| 模型 | YOLOv8n (yolov8n.pt 预训练权重) |
| 数据集 | GTSRB 43类 |
| 输入尺寸 | 640 × 640 |
| Batch Size | 8 |
| Epochs | 50 |
| 优化器 | 默认 (SGD/Adam) |
| 设备 | GPU (单卡) |

#### 训练结果

| 指标 | 数值 |
|------|------|
| mAP50 | 0.172 |
| mAP50-95 | 0.087 |
| Precision | ~0.17 |
| Recall | ~0.19 |
| F1-Score | 0.19 |

#### 分析

- ❌ **模型未收敛到正常水平**。GTSRB 数据集上 YOLOv8n 的 mAP50 预期应在 0.80+，当前结果远低于预期。
- 可能原因（待排查）：
  1. 数据路径或 `data.yaml` 配置问题（标签与图片未正确匹配）
  2. 标注格式转换错误（归一化计算有误）
  3. 类别 ID 超出 0~42 范围
  4. 训练超参数（学习率、batch size）不适合当前数据
- 📌 **结论**: 此 baseline 结果无效，需排查数据 pipeline 后重新训练。

---

### Exp-02: YOLOv8n + CBAM（待进行）

- 状态: 🔜 待 Baseline 修复后执行
- 计划: 在 Backbone 第3层（P3）插入 CBAM 注意力模块
- 预期: 提升小目标（远处交通标志）的召回率

## 📝 待办事项

- [ ] 排查 Exp-01 数据问题，重新训练 Baseline
- [ ] 实现 CBAM 模块并注册到本地 ultralytics
- [ ] 对比 Baseline vs CBAM 的 mAP、混淆矩阵
- [ ] 撰写实验分析报告

