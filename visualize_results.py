"""
建筑提取结果 - 增强可视化
生成更清晰直观的对比图
"""

import os
import numpy as np
import rasterio
from skimage import exposure, morphology
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

print("读取影像和提取结果...")

# 读取原图
with rasterio.open(os.path.join(SCRIPT_DIR, "D3C1207-300291A040_e.tif")) as src:
    img = src.read(1).astype(np.float32)
img = np.nan_to_num(img, nan=0)
p2, p98 = np.percentile(img, (2, 98))
img_norm = exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))

# 读取三种方法的结果
with rasterio.open(os.path.join(SCRIPT_DIR, "building_method1_threshold.tif")) as src:
    mask1 = src.read(1) > 0
with rasterio.open(os.path.join(SCRIPT_DIR, "building_method2_edge.tif")) as src:
    mask2 = src.read(1) > 0
with rasterio.open(os.path.join(SCRIPT_DIR, "building_method3_texture.tif")) as src:
    mask3 = src.read(1) > 0

# 取一块有代表性的区域进行展示
h, w = img_norm.shape
crop_size = 1500

# 相对中心平移（你可以改）
dy = 1000   # 向下移动多少像素（正=下，负=上）
dx = 2000   # 向右移动多少像素（正=右，负=左）

y_start = h // 2 - crop_size // 2 + dy
x_start = w // 2 - crop_size // 2 + dx

# 边界保护：不让窗口超出图像范围
y_start = max(0, min(y_start, h - crop_size))
x_start = max(0, min(x_start, w - crop_size))

y_end = y_start + crop_size
x_end = x_start + crop_size


print(f"显示区域: [{y_start}:{y_end}, {x_start}:{x_end}]")

# 裁剪
img_crop = img_norm[y_start:y_end, x_start:x_end]
mask1_crop = mask1[y_start:y_end, x_start:x_end]
mask2_crop = mask2[y_start:y_end, x_start:x_end]
mask3_crop = mask3[y_start:y_end, x_start:x_end]


def create_highlight_image(gray_img, mask, color='red'):
    """在灰度图上用颜色高亮标记区域"""
    # 转为RGB
    rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)
    
    # 提取建筑轮廓边界
    from skimage import segmentation
    boundary = segmentation.find_boundaries(mask, mode='thick')
    
    # 颜色映射
    colors = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0.5, 1],
        'yellow': [1, 1, 0],
        'magenta': [1, 0, 1],
        'cyan': [0, 1, 1]
    }
    c = colors.get(color, [1, 0, 0])
    
    # 填充区域用半透明颜色
    for i in range(3):
        rgb[:,:,i] = np.where(mask, rgb[:,:,i] * 0.5 + c[i] * 0.5, rgb[:,:,i])
    
    # 边界用实线颜色
    for i in range(3):
        rgb[:,:,i] = np.where(boundary, c[i], rgb[:,:,i])
    
    return np.clip(rgb, 0, 1)


# ===== 生成对比图1：三种方法并排对比 =====
print("\n生成对比图1：三种方法并排对比...")
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

# 原图
ax = axes[0, 0]

ax.imshow(img_crop, cmap='gray')
ax.set_title('原始遥感影像', fontsize=16, fontweight='bold')
ax.axis('off')

# 方法一
ax = axes[0, 1]
ax.imshow(create_highlight_image(img_crop, mask1_crop, 'red'))
ax.set_title('方法一：阈值分割法\n（红色=检测到的建筑）', fontsize=16, fontweight='bold')
ax.axis('off')

# 方法二
ax = axes[1, 0]
ax.imshow(create_highlight_image(img_crop, mask2_crop, 'green'))
ax.set_title('方法二：边缘检测+形态学\n（绿色=检测到的建筑）', fontsize=16, fontweight='bold')
ax.axis('off')

# 方法三
ax = axes[1, 1]
ax.imshow(create_highlight_image(img_crop, mask3_crop, 'cyan'))
ax.set_title('方法三：纹理特征(LBP)\n（青色=检测到的建筑）', fontsize=16, fontweight='bold')
ax.axis('off')

plt.suptitle('1973年北京遥感影像建筑提取 - 三种方法对比', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "result_comparison.png"), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  已保存: result_comparison.png")


# ===== 生成对比图2：每种方法单独的原图vs结果 =====
print("\n生成对比图2：单独方法对比...")

def save_method_comparison(mask_crop, method_name, color, filename):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左边原图
    axes[0].imshow(img_crop, cmap='gray')
    axes[0].set_title('原始影像', fontsize=18, fontweight='bold')
    axes[0].axis('off')
    
    # 右边提取结果
    axes[1].imshow(create_highlight_image(img_crop, mask_crop, color))
    axes[1].set_title(f'{method_name}提取结果\n（彩色区域=建筑）', fontsize=18, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(f'建筑提取 - {method_name}', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {filename}")

save_method_comparison(mask1_crop, '方法一：阈值分割法', 'red', 'result_method1.png')
save_method_comparison(mask2_crop, '方法二：边缘检测+形态学', 'green', 'result_method2.png')
save_method_comparison(mask3_crop, '方法三：纹理特征(LBP)', 'cyan', 'result_method3.png')


# ===== 生成仅建筑掩膜图 =====
print("\n生成纯建筑掩膜图...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

axes[0].imshow(mask1_crop, cmap='Reds')
axes[0].set_title('方法一：阈值分割\n白色=建筑', fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(mask2_crop, cmap='Greens')
axes[1].set_title('方法二：边缘+形态学\n白色=建筑', fontsize=16, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(mask3_crop, cmap='Blues')
axes[2].set_title('方法三：纹理LBP\n白色=建筑', fontsize=16, fontweight='bold')
axes[2].axis('off')

plt.suptitle('建筑提取掩膜对比（白色/彩色区域 = 检测到的建筑）', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "result_masks.png"), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  已保存: result_masks.png")


# ===== 统计汇总 =====
print("\n" + "=" * 50)
print("提取结果统计（裁剪区域）")
print("=" * 50)
total = mask1_crop.size
print(f"区域总像素: {total:,}")
print(f"方法一建筑: {mask1_crop.sum():,} ({100*mask1_crop.sum()/total:.1f}%)")
print(f"方法二建筑: {mask2_crop.sum():,} ({100*mask2_crop.sum()/total:.1f}%)")
print(f"方法三建筑: {mask3_crop.sum():,} ({100*mask3_crop.sum()/total:.1f}%)")

print("\n✅ 所有可视化图已生成完成！")
print("=" * 50)
