# Image Quality Evaluation

使用 Google Gemini API 并行评估模拟图像质量的工具。

## 特性

✅ **多 API 并行处理**：使用 3 个 API key 并发评估，大幅提升速度  
✅ **强大的错误处理**：API 错误不会中断程序，自动重试  
✅ **自动恢复**：支持断点续传，随时可以中断和恢复  
✅ **定期保存**：每处理 N 张图片自动保存进度  
✅ **详细日志**：记录所有错误信息到单独文件  
✅ **线程安全**：使用锁保护共享资源  
✅ **统计摘要**：自动生成评分分布统计

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：
- `google-genai` - Google Gemini API 客户端
- `Pillow` - 图片处理
- `tqdm` - 进度条（可选）

## 使用方法

### 方式 1：使用脚本（推荐）

```bash
cd /path/to/Taxonomy/scripts/image_render/image_process_quality_filter
bash run.sh
```

### 方式 2：直接运行 Python

使用默认的 3 个 API keys：

```bash
python evaluate_image_quality.py
```

使用自定义 API keys：

```bash
python evaluate_image_quality.py --api_keys KEY1 KEY2 KEY3
```

### 测试模式（只评估前 100 张图片）

```bash
python evaluate_image_quality.py --max_images 100
```

### 调整并行度

```bash
python evaluate_image_quality.py --num_workers 5
```

### 重新开始（忽略已有结果）

```bash
python evaluate_image_quality.py --no-resume
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_dir` | `/path/to/Taxonomy/Data/simulationImage` | 图片根目录 |
| `--output` | `./image_quality_ratings.json` | 输出 JSON 文件 |
| `--api_keys` | 使用内置的 3 个 keys | 自定义 API keys（空格分隔）|
| `--model` | `gemini-2.0-flash-exp` | 使用的 Gemini 模型 |
| `--num_workers` | `3` | 并行线程数 |
| `--save_interval` | `10` | 每 N 张图保存一次 |
| `--max_images` | `None` | 最大处理图片数（测试用）|
| `--image_name` | `lit.png` | 要评估的图片文件名 |
| `--resume` | `True` | 从已有结果继续 |
| `--no-resume` | - | 从头开始 |

## 评估标准

该工具评估四个维度，每个维度 0-10 分：

1. **Scene richness（场景丰富度）**：场景中可见的不同物体或元素数量
2. **Camera composition（相机构图）**：相机取景是否自然、平衡、视觉上令人愉悦
3. **Lighting and exposure（光照和曝光）**：光照是否自然，曝光是否平衡
4. **Rendering realism / clarity（渲染真实感/清晰度）**：纹理、反射和几何体是否真实和清晰

最终分数是四个子分数的平均值。

## 输出格式

### 主结果文件 (`image_quality_ratings.json`)

```json
{
  "/path/to/image.png": {
    "SceneRichness": 8.5,
    "Composition": 7.8,
    "LightingExposure": 9.0,
    "RealismClarity": 8.2,
    "FinalScore": 8.4,
    "Justification": "Well-composed scene with natural lighting and realistic details.",
    "user_dir": "jiawei",
    "scene_name": "1940Office",
    "room_id": "l000_r001",
    "full_path": "/path/to/image.png",
    "timestamp": "2024-11-06T15:30:00",
    "attempt": 1
  }
}
```

### 错误日志文件 (`image_quality_ratings_errors.json`)

```json
{
  "/path/to/failed_image.png": {
    "error": "ResourceExhausted: 429 Too Many Requests",
    "timestamp": "2024-11-06T15:30:00"
  }
}
```

### 统计摘要文件 (`image_quality_ratings_summary.json`)

```json
{
  "total_images": 1000,
  "mean_score": 7.8,
  "min_score": 3.2,
  "max_score": 9.8,
  "score_distribution": {
    "excellent (8-10)": 450,
    "good (6-8)": 380,
    "fair (4-6)": 150,
    "poor (0-4)": 20
  }
}
```

## 工作原理

### 多 API 并行处理

1. 脚本初始化 3 个 Gemini API 客户端
2. 使用线程池（默认 3 个线程）并行处理图片
3. 每个线程轮流使用不同的 API client（round-robin）
4. 这样可以绕过单个 API key 的速率限制

### 错误处理

- 每张图片最多重试 3 次
- 失败时自动切换到下一个 API client
- 所有错误都记录到错误日志文件
- 错误不会中断整个程序

### 断点续传

- 每处理 N 张图片（默认 10）自动保存进度
- 启动时自动加载已有结果
- 只处理未完成的图片
- 可随时中断（Ctrl+C）并稍后恢复

## 示例运行输出

```
Using 3 API keys
Initialized 3 API clients
Initializing evaluator with model: gemini-2.0-flash-exp
Found 5000 images to evaluate

============================================================
Total images: 5000
Already processed: 1200
To process: 3800
Workers: 3
============================================================

✓ [1/3800] Score: 8.2 - l000_r001/lit.png
✓ [2/3800] Score: 7.5 - l000_r002/lit.png
✓ [3/3800] Score: 9.1 - l000_r003/lit.png
...
→ Checkpoint saved (1210 results, 2 errors)
...
============================================================
Evaluation complete!
Total processed: 3800
Successful: 3795
Failed: 5
Success rate: 99.9%
Total in database: 5000
Results saved to: image_quality_ratings.json
Errors saved to: image_quality_ratings_errors.json
============================================================
```

## 故障排除

### 问题：API 配额错误

**症状**：大量 `ResourceExhausted: 429` 错误

**解决方案**：
- 减少并行度：`--num_workers 2`
- 使用更多 API keys：`--api_keys KEY1 KEY2 KEY3 KEY4`

### 问题：内存不足

**症状**：程序崩溃或变慢

**解决方案**：
- 减少并行度：`--num_workers 1`
- 分批处理：`--max_images 1000`

### 问题：想查看失败的图片

```bash
# 查看错误日志
cat image_quality_ratings_errors.json | jq 'keys'
```

## 性能优化建议

- 使用 SSD 存储图片（I/O 密集型任务）
- 增加 API keys 数量以提高并行度
- 在服务器上运行（更稳定的网络）
- 使用 `screen` 或 `tmux` 防止 SSH 断线

## 许可

使用需遵守 Google Gemini API 使用条款。
