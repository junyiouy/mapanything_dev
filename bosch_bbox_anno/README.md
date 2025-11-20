# Bosch Point Cloud Rendering and Bounding Box Annotation System

这个系统为 Bosch COLMAP 数据提供完整的点云渲染和边界框标注解决方案，基于 MapAnything 推理结果。

## 功能特性

- **相机参数提取**：从 GLB 文件或 MapAnything 预测结果中自动提取相机内外参
- **点云渲染**：使用 pyrender 从多个相机视角渲染点云
- **边界框标注**：交互式 3D 边界框标注工具
- **批量处理**：支持批量渲染多个场景
- **Gradio 集成**：可集成到现有的 Gradio Web 界面中

## 文件结构

```
bosch_bbox_anno/
├── camera_utils.py          # 相机参数提取和处理工具
├── pointcloud_renderer.py   # 点云渲染引擎和边界框可视化
├── bbox_annotator_ui.py     # 交互式标注界面
├── main_renderer.py         # 主渲染脚本和命令行工具
├── integrate_with_gradio.py # Gradio 应用集成
├── test_system.py          # 系统测试脚本
└── README.md               # 本文档
```

## 安装依赖

确保安装了以下 Python 包：

```bash
pip install numpy trimesh pyrender matplotlib gradio pillow
```

## 快速开始

### 1. 基本渲染

```bash
# 从 MapAnything 预测结果渲染点云
python main_renderer.py --input /path/to/predictions.npz --output_dir ./output

# 从 GLB 文件渲染
python main_renderer.py --glb_file /path/to/scene.glb --output_dir ./output
```

### 2. 带边界框的渲染

```bash
# 使用现有的边界框标注文件
python main_renderer.py --input predictions.npz --bbox_file bboxes.json --output_dir ./output

# 自动生成示例边界框
python main_renderer.py --input predictions.npz --create_sample_bboxes --output_dir ./output
```

### 3. 批量处理

```bash
# 批量渲染目录中的所有场景
python main_renderer.py --input_dir /path/to/scenes --batch_render --output_dir ./batch_output
```

### 4. 交互式标注

```bash
# 启动交互式边界框标注界面
python main_renderer.py --launch_annotator --input predictions.npz
```

## 命令行选项

### 主渲染脚本 (`main_renderer.py`)

- `--input, -i`: 输入文件 (NPZ 或 GLB 格式)
- `--input_dir`: 输入目录 (批量处理)
- `--glb_file`: GLB 场景文件
- `--output_dir, -o`: 输出目录 (默认: ./bosch_output)
- `--bbox_file`: 边界框标注 JSON 文件
- `--bbox_dir`: 边界框文件目录 (批量处理)
- `--create_sample_bboxes`: 自动生成示例边界框
- `--render_width`: 渲染图像宽度 (默认: 1280)
- `--render_height`: 渲染图像高度 (默认: 720)
- `--extract_cameras`: 仅提取相机参数
- `--launch_annotator`: 启动交互式标注界面
- `--batch_render`: 批量渲染多个场景

## API 使用

### 相机参数提取

```python
from camera_utils import extract_camera_parameters_from_predictions

# 从预测结果提取相机参数
cameras = extract_camera_parameters_from_predictions("predictions.npz")
print(f"提取了 {len(cameras)} 个相机配置")
```

### 点云渲染

```python
from pointcloud_renderer import PointCloudRenderer

# 创建渲染器
renderer = PointCloudRenderer(cameras, width=1280, height=720)

# 添加点云
renderer.add_point_cloud(points, colors)

# 添加边界框
bboxes = [
    {'min': [0, 0, 0], 'max': [1, 1, 1], 'label': 'object_1'},
    {'min': [2, 2, 2], 'max': [3, 3, 3], 'label': 'object_2'}
]
renderer.add_bounding_boxes(bboxes)

# 从相机 0 渲染
image = renderer.render_from_camera(0, show_bboxes=True)

# 渲染所有相机视角
renderer.render_all_cameras("./output", "scene")
```

### 边界框标注

```python
from pointcloud_renderer import BoundingBoxAnnotator

# 创建标注器
annotator = BoundingBoxAnnotator(points, colors)

# 手动添加边界框
annotator.add_bbox_manual(
    min_bounds=np.array([0, 0, 0]),
    max_bounds=np.array([1, 1, 1]),
    label="my_object"
)

# 保存标注
annotator.save_annotations("bboxes.json")

# 获取统计信息
stats = annotator.get_bbox_statistics()
```

## Gradio 集成

要将此系统集成到现有的 Gradio 应用中：

```python
from bosch_bbox_anno.integrate_with_gradio import add_pointcloud_rendering_tab

# 在您的 gradio_app.py 中，创建应用后添加：
gradio_app = add_pointcloud_rendering_tab(gradio_app, target_dir_state)
```

## 输出格式

### 相机参数 JSON

```json
[
  {
    "name": "camera_0",
    "intrinsics": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]],
    "extrinsics": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "camera_model": "pinhole"
  }
]
```

### 边界框标注 JSON

```json
[
  {
    "min": [0.0, 0.0, 0.0],
    "max": [1.0, 1.0, 1.0],
    "label": "object_1",
    "num_points": 150
  }
]
```

## 工作流程示例

1. **运行 MapAnything 推理**：
   ```bash
   python scripts/infer_bosch_data_chunk_new_data.py --data_root /path/to/data --output_dir ./results
   ```

2. **提取相机参数**：
   ```bash
   python bosch_bbox_anno/main_renderer.py --input results/predictions.npz --extract_cameras --output_dir ./results
   ```

3. **交互式标注**：
   ```bash
   python bosch_bbox_anno/main_renderer.py --launch_annotator --input results/predictions.npz
   ```

4. **渲染带标注的场景**：
   ```bash
   python bosch_bbox_anno/main_renderer.py --input results/predictions.npz --bbox_file results/bounding_boxes.json --output_dir ./final_output
   ```

## 故障排除

### 常见问题

1. **GLB 文件加载失败**
   - 确保 GLB 文件包含有效的点云几何体
   - 检查 trimesh 版本兼容性

2. **渲染失败**
   - 确保安装了 pyrender 和相关依赖
   - 检查相机参数的格式是否正确

3. **内存不足**
   - 减小渲染分辨率 (`--render_width`, `--render_height`)
   - 对于大型点云，考虑降采样

### 依赖问题

如果遇到 pyrender 相关问题，可以尝试：

```bash
# 安装系统依赖 (Ubuntu)
sudo apt-get install libegl1-mesa libgl1-mesa-glx

# 或使用软件渲染
export PYOPENGL_PLATFORM=osmesa
```

## 许可证

本项目遵循 Apache License 2.0 许可证。
