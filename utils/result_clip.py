import os
import numpy as np
import rasterio
from PIL import Image

def clip_result_with_mask(pred_path, mask_path, save_path):
    print(f"Processing: {os.path.basename(pred_path)}...")
    
    # Check if files exist
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} does not exist.")
        return
    if not os.path.exists(mask_path):
        print(f"Error: {mask_path} does not exist.")
        return

    # Read Prediction TIF
    with rasterio.open(pred_path) as src_pred:
        pred_data = src_pred.read(1) # Read first band
        profile = src_pred.profile

    # Read ROI Mask TIF
    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        
    # Ensure shapes match
    if pred_data.shape != mask_data.shape:
        print(f"  Warning: Shape Mismatch: Pred {pred_data.shape} vs Mask {mask_data.shape}. Resizing mask...")
        # Resize mask to match pred using PIL
        mask_img = Image.fromarray(mask_data)
        mask_img = mask_img.resize((pred_data.shape[1], pred_data.shape[0]), resample=Image.NEAREST)
        mask_data = np.array(mask_img)

    # Create destination profile
    profile.update({
        'driver': 'GTiff',
        'height': pred_data.shape[0],
        'width': pred_data.shape[1],
        'alpha': 'no' 
    })
    
    # Determine NoData value
    dtype = pred_data.dtype
    nodata_val = None
    
    if np.issubdtype(dtype, np.floating):
        nodata_val = np.nan
    elif dtype == np.uint8:
        nodata_val = 254
    else:
        nodata_val = -9999
        
    profile.update(nodata=nodata_val)
    
    # Apply Mask: Set outside ROI to nodata_val
    valid_mask = (mask_data == 1)
    
    out_data = pred_data.copy()
    # 将水体设为 255 (白色)
    out_data[out_data == 1] = 255
    
    # Fill Nodata
    if nodata_val is not None:
         out_data[~valid_mask] = nodata_val
         
    # Save as TIF
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(out_data, 1)
        
    print(f"  -> Saved clipped result to: {os.path.basename(save_path)}")

    # [新增] 保存为 PNG (带透明通道)
    png_save_path = os.path.splitext(save_path)[0] + ".png"
    
    h, w = out_data.shape
    # 创建 RGBA 图像 (默认为全黑全透明)
    rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
    
    if dtype == np.uint8:
        # 逻辑: 0->黑色, 255->白色, 254(NoData)->透明
        
        # 1. 设置 Alpha 通道 (透明度)
        # 凡是不等于 NoData (254) 的地方，Alpha = 255 (完全不透明)
        valid_pixels = (out_data != nodata_val)
        rgba_img[valid_pixels, 3] = 255
        
        # 2. 设置 RGB 通道 (颜色)
        # 水体 (255) -> 白色 (255, 255, 255)
        water_pixels = (out_data == 255)
        rgba_img[water_pixels, 0] = 255
        rgba_img[water_pixels, 1] = 255
        rgba_img[water_pixels, 2] = 255
        
        # 背景 (0) -> 黑色 (0, 0, 0) 
        # (初始化时已经是0了，所以不用额外操作)
        
    elif np.issubdtype(dtype, np.floating):
        # 逻辑: 0.0-1.0 -> 0-255灰度, NaN -> 透明
        
        # 处理有效区域
        valid_pixels = ~np.isnan(out_data)
        rgba_img[valid_pixels, 3] = 255
        
        # 归一化并填充 RGB
        # 假设数据在 0-1 之间，映射到 0-255
        val_scaled = np.clip(out_data, 0, 1) * 255
        val_scaled = np.nan_to_num(val_scaled).astype(np.uint8)
        
        rgba_img[..., 0] = val_scaled
        rgba_img[..., 1] = val_scaled
        rgba_img[..., 2] = val_scaled

    # 保存 PNG
    Image.fromarray(rgba_img).save(png_save_path)
    print(f"  -> Saved PNG preview to: {os.path.basename(png_save_path)}")

if __name__ == "__main__":
    base_dir = "/home/rove/lancing"
    infer_dir = os.path.join(base_dir, "lancang_infer")
    save_dir = os.path.join(infer_dir, "clip")
    
    # [新增] 自动创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    mask_filename = "roi_mask.tif"
    mask_path = os.path.join(infer_dir, mask_filename)
    
    if not os.path.exists(infer_dir):
        print(f"Error: Directory {infer_dir} not found.")
        exit(1)
        
    if not os.path.exists(mask_path):
        print(f"Error: Mask file {mask_path} not found.")
        exit(1)

    print(f"Scanning directory: {infer_dir}")
    
    # 获取目录下所有文件
    files = os.listdir(infer_dir)
    clip_files = os.listdir(save_dir) if os.path.exists(save_dir) else []
    
    count = 0
    for f in files:
        # 1. 必须是 tif 文件
        if not f.endswith(".tif"):
            continue
            
        # 2. 跳过 mask 文件本身
        if f == mask_filename:
            continue
            
        # 3. 跳过已经 clip 过的文件 (防止重复处理)
        if f.replace(".tif", "_clip.tif") in clip_files:
            print(f"  Skipping already clipped file: {f}")
            continue
            
        # 构造路径
        pred_file = os.path.join(infer_dir, f)
        save_file = os.path.join(save_dir, f.replace(".tif", "_clip.tif"))
        
        try:
            clip_result_with_mask(pred_file, mask_path, save_file)
            count += 1
        except Exception as e:
            print(f"  Error processing {f}: {e}")
