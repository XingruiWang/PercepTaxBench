# Scene Structure Annotation Generator

从分割掩码生成场景结构标注（2D/3D边界框、对象列表等）。

## 功能

- 📦 从分割掩码提取对象边界框（2D 和 3D）
- 🎨 生成可视化图片（bbox_visualization_all.png, bbox_3d_visualization.png）
- 📝 输出结构化 JSON 标注文件（scene_annotations_split.json）
- 🚀 支持批量处理整个数据集
- ⚡ 多线程并行处理

## 使用方法

### 方式 1：批量处理所有场景（推荐）

```bash
cd /path/to/Taxonomy/scripts/image_render/image_annotation
bash run_batch_annotation.sh
```

这将处理 `simulationImage` 目录下的所有场景，生成标注到：
`/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations`

### 方式 2：使用 Python 直接运行

**处理整个数据集：**
```bash
python gen_scene_structure.py \
    /path/to/Taxonomy/Data/simulationImage \
    --batch \
    --visualize \
    --num_workers 4
```

**处理单个场景：**
```bash
python gen_scene_structure.py \
    /path/to/simulationImage/jiawei/1940Office/l000_r001 \
    --visualize
```

### 方式 3：从 image_quality_ratings.json 处理（质量过滤）

当质量评分文件准备好后：
```bash
python gen_scene_structure.py \
    --use_json \
    --from_json /path/to/image_quality_ratings.json \
    --num_workers 4 \
    --visualize
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `simulationImage` 路径 | 输入数据目录 |
| `--output_dir` | `SimulationMetadata/scenes/annotations` | 输出目录 |
| `--batch` | False | 批量处理模式 |
| `--visualize` | False | 生成可视化图片 |
| `--num_workers` | 1 | 并行线程数 |
| `--skip_3d` | False | 跳过 3D bbox 计算 |
| `--use_json` | False | 从 JSON 文件读取场景列表 |
| `--from_json` | 路径 | 质量评分 JSON 文件路径 |

## 输出结构

对于每个场景（例如 `1940Office/l000_r001`），会生成：

```
annotations/
└── 1940Office/
    └── l000_r001/
        ├── scene_annotations_split.json    # 结构化标注
        ├── bbox_visualization_all.png      # 2D bbox 可视化
        └── bbox_3d_visualization.png       # 3D bbox 可视化
```

### scene_annotations_split.json 格式

```json
{
  "scene_name": "1940Office",
  "room_id": "l000_r001",
  "image_path": "/path/to/lit.png",
  "objects": [
    {
      "class_name": "chair",
      "instance_id": "chair_001",
      "bbox_2d": [x1, y1, x2, y2],
      "bbox_3d": {
        "center": [x, y, z],
        "size": [w, h, d],
        "rotation": angle
      },
      "pixel_count": 1234,
      "mask_area": 0.05
    }
  ],
  "statistics": {
    "total_objects": 15,
    "unique_classes": 8
  }
}
```

## 预计处理时间

- **单个场景**：~2-5 秒
- **整个数据集**（~5000 个场景）：
  - 单线程：~4-6 小时
  - 4 线程：~1-2 小时
  - 8 线程：~30-60 分钟

## 依赖要求

```bash
pip install numpy opencv-python pillow scipy
```

## 并行处理建议

```bash
# 轻度：2 个线程（稳定）
python gen_scene_structure.py --batch --num_workers 2

# 中度：4 个线程（推荐）
python gen_scene_structure.py --batch --num_workers 4

# 重度：8 个线程（需要足够内存）
python gen_scene_structure.py --batch --num_workers 8
```

## 工作流程

### 当前阶段：生成基础标注

```bash
# Step 1: 生成所有场景的标注
cd /path/to/Taxonomy/scripts/image_render/image_annotation
bash run_batch_annotation.sh
```

### 未来阶段：基于质量过滤

```bash
# Step 1: 图片质量评分（正在进行中）
cd /path/to/Taxonomy/scripts/image_render/image_process_quality_filter
bash run.sh

# Step 2: 基于质量评分生成标注
cd /path/to/Taxonomy/scripts/image_render/image_annotation
python gen_scene_structure.py \
    --use_json \
    --from_json /path/to/image_quality_ratings.json \
    --num_workers 4
```

## 故障排除

### 问题 1：找不到分割掩码

**症状**：`FileNotFoundError: seg.png not found`

**解决**：确保输入目录包含 `seg.png` 文件

### 问题 2：内存不足

**症状**：程序崩溃或变慢

**解决**：
```bash
# 减少并行度
python gen_scene_structure.py --batch --num_workers 1
```

### 问题 3：某些场景处理失败

**症状**：部分场景没有生成标注

**解决**：
- 检查 `processing_errors.txt` 文件
- 单独处理失败的场景进行调试

## 输出文件说明

- **scene_annotations_split.json**：完整的结构化标注
- **bbox_visualization_all.png**：所有对象的 2D 边界框可视化
- **bbox_3d_visualization.png**：3D 边界框投影可视化
- **processing_errors.txt**：处理错误日志

## 与其他模块的关系

```
simulationImage/          # 输入：原始场景数据
    └── */lit.png, seg.png

    ↓ gen_scene_structure.py

SimulationMetadata/
└── scenes/
    ├── annotations/      # 输出：标注文件
    └── image_quality_ratings.json  # 质量评分（可选）
```

## 后续步骤

1. ✅ 生成所有场景的标注
2. ⏳ 完成图片质量评分
3. 📊 根据质量分数过滤场景
4. 🚀 上传到 Hugging Face

