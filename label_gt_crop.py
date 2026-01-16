import os
import numpy as np
import rasterio
from skimage import exposure
import napari

# ========= 配置 =========
BASE = r"/Users/libingchen/Desktop/KH_1973 (2)"
IMG_PATH = os.path.join(BASE, "D3C1207-300291A040_e.tif")

# 裁剪窗口大小（建议 1000~2000）
CROP_SIZE = 1500

# 输出：保存真值mask + 保存窗口坐标（便于复现/算IoU）
OUT_MASK = os.path.join(BASE, "gt_mask_crop.npy")
OUT_META = os.path.join(BASE, "gt_mask_crop_meta.npy")

print("START label_gt_crop.py")

# ========= 读取原图并做同样的归一化 =========
with rasterio.open(IMG_PATH) as src:
    img = src.read(1).astype(np.float32)

img = np.nan_to_num(img, nan=0)
p2, p98 = np.percentile(img, (2, 98))
img_norm = exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))

# ========= 取中心裁剪（你也可以换成别的窗口） =========
h, w = img_norm.shape
y_start = h // 2 - CROP_SIZE // 2
x_start = w // 2 - CROP_SIZE // 2

# 边界保护
y_start = max(0, min(y_start, h - CROP_SIZE))
x_start = max(0, min(x_start, w - CROP_SIZE))

y_end = y_start + CROP_SIZE
x_end = x_start + CROP_SIZE

crop = img_norm[y_start:y_end, x_start:x_end]

# 初始化 labels（0=背景，1=建筑）
labels = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

print("=== napari 标注说明 ===")
print("1) 左侧选择 GroundTruth 图层（labels）")
print("2) Label value 设为 1，用画笔涂建筑")
print("3) 擦除：Label value 设为 0 再涂")
print("4) 完成后：直接关闭窗口，会自动保存 gt_mask_crop.npy")
print(f"窗口坐标: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")

viewer = napari.Viewer()
viewer.add_image(crop, name="Image (crop)", colormap="gray", contrast_limits=(0, 1))
labels_layer = viewer.add_labels(labels, name="GroundTruth (0=bg,1=bldg)")

# 关闭窗口时自动保存
def _save_and_close():
    np.save(OUT_MASK, labels_layer.data.astype(np.uint8))
    np.save(OUT_META, np.array([y_start, y_end, x_start, x_end], dtype=np.int64))
    print(f"✅ 已保存真值mask: {OUT_MASK}")
    print(f"✅ 已保存窗口坐标: {OUT_META}")

# 绑定关闭事件（Qt）
qt_window = viewer.window._qt_window
old_close = qt_window.closeEvent
def new_close_event(e):
    _save_and_close()
    e.accept()
qt_window.closeEvent = new_close_event

print("Launching napari window...")
napari.run()
