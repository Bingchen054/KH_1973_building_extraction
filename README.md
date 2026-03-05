# 1973年北京遥感影像提取

基于Python的历史遥感影像建筑物自动提取工具，实现三种不同的提取方法进行对比。

## 📁 项目结构

```
KH_1973/
├── D3C1207-300291A040_e.tif    # 1973年北京遥感原始影像
├── building_extraction.py       # 主程序：三种建筑提取方法
├── visualize_results.py         # 可视化脚本
├── building_method1_threshold.tif  # 方法一结果
├── building_method2_edge.tif       # 方法二结果
├── building_method3_texture.tif    # 方法三结果
├── result_comparison.png        # 三种方法对比图
├── result_method1.png           # 方法一单独对比
├── result_method2.png           # 方法二单独对比
├── result_method3.png           # 方法三单独对比
└── result_masks.png             # 纯建筑掩膜图
```

## 🔧 环境配置

```bash
conda activate chepai
pip install rasterio scikit-image scipy matplotlib numpy
```

## 🚀 使用方法

### 1. 运行建筑提取
```bash
python building_extraction.py
```

### 2. 生成可视化结果
```bash
python visualize_results.py
```

## 📊 三种提取方法

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| **方法一：阈值分割** | Otsu自动阈值，提取高反射率区域 | 建筑屋顶较亮的区域 |
| **方法二：边缘检测+形态学** | Canny边缘检测 + 区域填充 | 边界清晰的规则建筑 |
| **方法三：纹理特征(LBP)** | 局部二值模式纹理 + 亮度联合判断 | 纹理明显的人工建筑 |

## 📈 结果示例

运行后会生成以下可视化文件：
- `result_comparison.png` - 四宫格对比图（原图 + 三种方法）
- `result_method*.png` - 各方法的原图vs结果对比
- `result_masks.png` - 纯建筑掩膜图（白色=建筑）


