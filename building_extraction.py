"""
1973年北京遥感影像建筑提取
============================================
方法一：阈值分割法 (Threshold Segmentation)
方法二：边缘检测+形态学法 (Edge Detection + Morphology)  
方法三：纹理特征法 (Texture Feature - LBP)
"""

import os
import numpy as np
import rasterio
from skimage import exposure, filters, morphology, feature
from skimage.feature import local_binary_pattern
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========= 配置 =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = r"/Users/libingchen/Desktop/KH_1973 (2)/D3C1207-300291A040_e.tif"
OUT_DIR  = r"/Users/libingchen/Desktop/KH_1973 (2)"


print("=" * 60)
print("1973年北京遥感影像建筑提取")
print("=" * 60)
print(f"影像路径: {IMG_PATH}")
print(f"输出目录: {OUT_DIR}")

# ========= 读取影像 =========
print("\n[1] 读取影像...")
with rasterio.open(IMG_PATH) as src:
    img = src.read(1).astype(np.float32)
    profile = src.profile
    
print(f"    影像尺寸: {img.shape}")
print(f"    数据类型: {img.dtype}")
print(f"    数值范围: [{img.min():.2f}, {img.max():.2f}]")

# 处理无效值
img = np.nan_to_num(img, nan=0)

# 对比度拉伸（2%-98%百分位）
p2, p98 = np.percentile(img, (2, 98))
img_norm = exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))
print(f"    对比度拉伸: [{p2:.2f}, {p98:.2f}] -> [0, 1]")


# ================================================================
# 方法一：阈值分割法
# ================================================================
def method1_threshold(img_norm):
    """
    基于阈值分割的建筑提取
    原理：建筑物屋顶通常具有较高的反射率（较亮）
    """
    print("\n[方法一] 阈值分割法...")
    
    # 使用Otsu自动阈值
    thresh_otsu = filters.threshold_otsu(img_norm)
    print(f"    Otsu阈值: {thresh_otsu:.4f}")
    
    # 建筑通常较亮，取亮区域
    building_mask = img_norm > thresh_otsu
    
    # 形态学清理
    # 去除小噪点
    building_mask = morphology.remove_small_objects(building_mask, min_size=200)
    # 填充建筑内部小孔
    building_mask = morphology.remove_small_holes(building_mask, area_threshold=500)
    # 闭运算平滑边界
    building_mask = morphology.closing(building_mask, morphology.disk(3))
    
    pixel_count = building_mask.sum()
    print(f"    检测到建筑像素: {pixel_count:,}")
    
    return building_mask.astype(np.uint8)


# ================================================================
# 方法二：边缘检测 + 形态学法
# ================================================================
def method2_edge_morphology(img_norm):
    """
    基于边缘检测和形态学的建筑提取
    原理：建筑物具有明显的边缘和规则的几何形状
    """
    print("\n[方法二] 边缘检测+形态学法...")
    
    # Canny边缘检测
    edges = feature.canny(img_norm, sigma=2, low_threshold=0.05, high_threshold=0.15)
    print(f"    边缘像素数: {edges.sum():,}")
    
    # 膨胀边缘，连接断裂的轮廓
    dilated = morphology.dilation(edges, morphology.disk(3))
    
    # 填充闭合区域
    filled = ndimage.binary_fill_holes(dilated)
    
    # 去除小区域（噪点）
    cleaned = morphology.remove_small_objects(filled, min_size=300)
    
    # 开运算去除细小连接
    opened = morphology.opening(cleaned, morphology.disk(2))
    
    # 闭运算填充小孔洞
    building_mask = morphology.closing(opened, morphology.disk(3))
    
    # 再次去除小区域
    building_mask = morphology.remove_small_objects(building_mask, min_size=500)
    
    pixel_count = building_mask.sum()
    print(f"    检测到建筑像素: {pixel_count:,}")
    
    return building_mask.astype(np.uint8)


# ================================================================
# 方法三：纹理特征法 (LBP)
# ================================================================
def method3_texture_lbp(img_norm):
    """
    基于局部二值模式(LBP)纹理特征的建筑提取
    原理：建筑物的纹理模式与自然地物(植被、水体)不同
    """
    print("\n[方法三] 纹理特征法(LBP)...")
    
    # 转为8位图像用于LBP计算
    img_8bit = (img_norm * 255).astype(np.uint8)
    
    # 计算LBP特征
    # radius=3, n_points=24 是常用配置
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_8bit, n_points, radius, method='uniform')
    print(f"    LBP参数: radius={radius}, n_points={n_points}")
    print(f"    LBP值范围: [{lbp.min():.2f}, {lbp.max():.2f}]")
    
    # 计算局部方差（纹理复杂度）
    # 建筑通常有中等复杂度的纹理
    from scipy.ndimage import uniform_filter, variance
    
    win_size = 9
    local_mean = uniform_filter(img_norm, size=win_size)
    local_sqr_mean = uniform_filter(img_norm**2, size=win_size)
    local_var = local_sqr_mean - local_mean**2
    local_var = np.clip(local_var, 0, None)  # 确保非负
    
    print(f"    局部方差范围: [{local_var.min():.6f}, {local_var.max():.6f}]")
    
    # 结合亮度和纹理进行分类
    # 建筑：较亮 + 中等纹理
    bright_mask = img_norm > np.percentile(img_norm, 60)
    
    # 纹理：排除非常平滑（水体）和非常粗糙（某些植被）
    var_low, var_high = np.percentile(local_var, (20, 90))
    texture_mask = (local_var > var_low) & (local_var < var_high)
    
    # 组合条件
    building_mask = bright_mask & texture_mask
    
    # 形态学清理
    building_mask = morphology.remove_small_objects(building_mask, min_size=300)
    building_mask = morphology.remove_small_holes(building_mask, area_threshold=500)
    building_mask = morphology.closing(building_mask, morphology.disk(3))
    building_mask = morphology.opening(building_mask, morphology.disk(2))
    
    pixel_count = building_mask.sum()
    print(f"    检测到建筑像素: {pixel_count:,}")
    
    return building_mask.astype(np.uint8)


# ========= 运行三种方法 =========
print("\n" + "=" * 60)
print("开始运行三种方法...")
print("=" * 60)

result1 = method1_threshold(img_norm)
result2 = method2_edge_morphology(img_norm)
result3 = method3_texture_lbp(img_norm)


# ========= 保存结果 =========
print("\n[2] 保存结果...")

def save_result(mask, name, profile):
    """保存结果为GeoTIFF"""
    out_path = os.path.join(OUT_DIR, f"building_{name}.tif")
    profile_copy = profile.copy()
    profile_copy.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(out_path, "w", **profile_copy) as dst:
        dst.write((mask * 255), 1)
    print(f"    已保存: {out_path}")
    return out_path

save_result(result1, "method1_threshold", profile)
save_result(result2, "method2_edge", profile)
save_result(result3, "method3_texture", profile)


# ========= 可视化对比 =========
print("\n[3] 生成可视化对比图...")

# 由于原图很大，取一个子区域进行可视化
h, w = img_norm.shape
# 取中心区域的一部分用于展示
crop_size = min(2000, h//2, w//2)
y_start = h // 2 - crop_size // 2
x_start = w // 2 - crop_size // 2
y_end = y_start + crop_size
x_end = x_start + crop_size

print(f"    可视化区域: [{y_start}:{y_end}, {x_start}:{x_end}]")

# 创建对比图
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# 原图
ax1 = axes[0, 0]
ax1.imshow(img_norm[y_start:y_end, x_start:x_end], cmap='gray')
ax1.set_title('原始影像', fontsize=14)
ax1.axis('off')

# 方法一结果
ax2 = axes[0, 1]
ax2.imshow(img_norm[y_start:y_end, x_start:x_end], cmap='gray')
ax2.imshow(result1[y_start:y_end, x_start:x_end], cmap='Reds', alpha=0.5)
ax2.set_title('方法一：阈值分割法', fontsize=14)
ax2.axis('off')

# 方法二结果
ax3 = axes[1, 0]
ax3.imshow(img_norm[y_start:y_end, x_start:x_end], cmap='gray')
ax3.imshow(result2[y_start:y_end, x_start:x_end], cmap='Greens', alpha=0.5)
ax3.set_title('方法二：边缘检测+形态学法', fontsize=14)
ax3.axis('off')

# 方法三结果
ax4 = axes[1, 1]
ax4.imshow(img_norm[y_start:y_end, x_start:x_end], cmap='gray')
ax4.imshow(result3[y_start:y_end, x_start:x_end], cmap='Blues', alpha=0.5)
ax4.set_title('方法三：纹理特征法(LBP)', fontsize=14)
ax4.axis('off')

plt.suptitle('1973年北京遥感影像建筑提取 - 三种方法对比', fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存对比图
comparison_path = os.path.join(OUT_DIR, "building_comparison.png")
plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"    已保存对比图: {comparison_path}")

plt.close()


# ========= 统计信息 =========
print("\n" + "=" * 60)
print("提取结果统计")
print("=" * 60)
total_pixels = img_norm.size
print(f"影像总像素: {total_pixels:,}")
print(f"方法一 (阈值分割):     {result1.sum():>12,} 像素 ({100*result1.sum()/total_pixels:.2f}%)")
print(f"方法二 (边缘+形态学):  {result2.sum():>12,} 像素 ({100*result2.sum()/total_pixels:.2f}%)")
print(f"方法三 (纹理LBP):      {result3.sum():>12,} 像素 ({100*result3.sum()/total_pixels:.2f}%)")

print("\n" + "=" * 60)
print("✅ 处理完成!")
print("=" * 60)
