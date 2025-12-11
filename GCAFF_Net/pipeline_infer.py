import torch
import numpy as np
import rasterio
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from config import config
from model import Model

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
# 1. Paths
MODEL_WEIGHTS_PATH = "checkpoints/GCAFF_Net_1204/best_model.pth" 
INFERENCE_FILE_PATH = "/mnt/data1/rove/dataset/S1_Water/infer/2412_rec_vv_vh.tif"
GT_PATH = "/mnt/data1/rove/dataset/S1_Water/infer/watermask2412.tif"
ROI_PATH = "/mnt/data1/rove/dataset/S1_Water/infer/roi_mask.tif"
RESULT_SAVE_PATH = f"./lancang_infer/lancang_river_{config['model_name']}.tif"

# 2. Normalization (Must match training)
NORM_MEAN = [-1148.2476, -1944.6511]
NORM_STD  = [594.0617, 973.9897]

# 3. Inference Settings
INFER_BATCH_SIZE = 256
INFER_NUM_WORKERS = 8

class SlidingWindowDataset(Dataset):
    def __init__(self, image_padded, rows, cols, patch_size, mean, std):
        self.image = image_padded
        self.rows = rows
        self.cols = cols
        self.patch_size = patch_size
        
        self.mean = torch.tensor(mean).view(-1, 1, 1).float()
        self.std = torch.tensor(std).view(-1, 1, 1).float()

    def __len__(self):
        return len(self.rows) * len(self.cols)

    def __getitem__(self, idx):
        r_idx = idx // len(self.cols)
        c_idx = idx % len(self.cols)
        
        r = self.rows[r_idx]
        c = self.cols[c_idx]
        
        patch = self.image[:, r:r+self.patch_size, c:c+self.patch_size]
        
        patch_tensor = torch.from_numpy(patch).float()
        patch_tensor = (patch_tensor - self.mean) / self.std
        
        return patch_tensor, r, c

def get_gaussian_mask(size, sigma_scale=1.0/8):
    tmp = np.zeros((size, size))
    center = size // 2
    sig = size * sigma_scale
    y, x = np.ogrid[-center:size-center, -center:size-center]
    mask = np.exp(-(x**2 + y**2) / (2 * sig**2))
    return mask

def predict_sliding_window(model, image_path, patch_size, stride, num_classes, mean, std, device):
    """
    Step 1: Full Image Inference
    Returns:
        prob_map (np.ndarray): The raw probability map (0-1) of shape (H, W).
        profile (dict): Rasterio profile for saving.
    """
    print(f"\n{'='*20} Step 1: Full Image Inference {'='*20}")
    print(f"Target File: {image_path}")
    
    # 1. Read Image & Preprocess
    with rasterio.open(image_path) as src:
        image = src.read() # (C, H, W)
        profile = src.profile
        image = image.astype(np.float32)
        
        print(f"Original Data Range: Min {np.nanmin(image):.2f}, Max {np.nanmax(image):.2f}")

        # --- Fix 1: Scale Adjustment (dB -> Scaled Int16 domain) ---
        # 训练数据看起来是 dB * 100，所以这里也要乘以 100
        print("Applying scaling factor 100.0 to match training distribution...")
        image = image * 100.0
        if np.isnan(image).any():
            print("Found NaN values. Filling with training mean...")
            for c_idx in range(image.shape[0]):
                fill_val = mean[c_idx] if c_idx < len(mean) else 0
                
                mask = np.isnan(image[c_idx])
                nan_count = np.sum(mask)
                
                if nan_count > 0:
                    image[c_idx][mask] = fill_val
                    print(f"  Channel {c_idx}: Filled {nan_count} NaN pixels with {fill_val}")
        
        c, h, w = image.shape
        print(f"Processed Size: {c}x{h}x{w}")
        print(f"Processed Data Range: Min {image.min():.2f}, Max {image.max():.2f}")
        
    # 2. Padding
    pad_h = stride - (h % stride) if h % stride != 0 else 0
    pad_w = stride - (w % stride) if w % stride != 0 else 0
    if h + pad_h < patch_size: pad_h = patch_size - h
    if w + pad_w < patch_size: pad_w = patch_size - w

    image_padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    _, h_pad, w_pad = image_padded.shape
    
    # 3. Coordinates
    rows = list(range(0, h_pad - patch_size + 1, stride))
    cols = list(range(0, w_pad - patch_size + 1, stride))
    
    # 4. DataLoader
    dataset = SlidingWindowDataset(image_padded, rows, cols, patch_size, mean, std)
    loader = DataLoader(
        dataset, 
        batch_size=INFER_BATCH_SIZE, 
        shuffle=False, 
        num_workers=INFER_NUM_WORKERS, 
        pin_memory=True
    )
    
    # 5. Result Containers (CPU)
    prob_map = torch.zeros((num_classes, h_pad, w_pad), dtype=torch.float32, device='cpu')
    weight_map = torch.zeros((h_pad, w_pad), dtype=torch.float32, device='cpu')
    
    gaussian_mask = get_gaussian_mask(patch_size)
    gaussian_mask_tensor = torch.from_numpy(gaussian_mask).float().to(device)
    
    model.eval()
    
    # 6. Inference
    print(f"Starting Inference (Batch Size: {INFER_BATCH_SIZE})...")
    with torch.no_grad():
        for batch_patches, batch_r, batch_c in tqdm(loader, desc="Inference Progress", unit="batch"):
            batch_patches = batch_patches.to(device)
            
            outputs = model(batch_patches)
            outputs = torch.sigmoid(outputs)
            outputs = outputs * gaussian_mask_tensor
            
            outputs_cpu = outputs.cpu() 
            
            for i in range(outputs_cpu.shape[0]):
                r = batch_r[i].item()
                c = batch_c[i].item()
                
                weighted_patch = outputs_cpu[i].squeeze(0) 
                prob_map[:, r:r+patch_size, c:c+patch_size] += weighted_patch
                weight_map[r:r+patch_size, c:c+patch_size] += gaussian_mask_tensor.cpu()

    # 7. Normalize & Crop
    weight_map = torch.where(weight_map == 0, torch.ones_like(weight_map), weight_map)
    prob_map /= weight_map
    prob_map = prob_map[:, :h, :w]
    
    # Return numpy array (H, W)
    return prob_map.squeeze(0).numpy(), profile

def apply_roi_mask_and_save(prob_map, roi_path, save_path, profile):
    """
    Step 2: ROI Masking & Saving
    """
    print(f"\n{'='*20} Step 2: ROI Masking & Saving {'='*20}")
    
    # Load ROI
    with rasterio.open(roi_path) as src:
        roi_mask = src.read(1) # (H, W)
        
    if roi_mask.shape != prob_map.shape:
        print(f"Warning: ROI shape {roi_mask.shape} != Prediction shape {prob_map.shape}. Resizing ROI...")
        # Simple resize if needed, but ideally they should match
        # For now assume they match or raise error
        raise ValueError(f"Shape mismatch: ROI {roi_mask.shape} vs Pred {prob_map.shape}")

    # Apply Mask: ROI==0 -> Background (0)
    # Threshold prediction first
    pred_binary = (prob_map > 0.3).astype(np.uint8)
    
    # Masking
    final_result = pred_binary * (roi_mask > 0).astype(np.uint8)
    
    # Save
    final_result_save = (final_result * 255).astype(np.uint8)
    
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(final_result_save, 1)
        
    print(f"Masked result saved to: {save_path}")
    return final_result

def evaluate_metrics(pred_binary, gt_path, roi_path):
    """
    Step 3: Accuracy Evaluation (Inside ROI Only)
    """
    print(f"\n{'='*20} Step 3: Accuracy Evaluation {'='*20}")
    
    # Load GT and ROI
    with rasterio.open(gt_path) as src:
        gt = src.read(1)
    with rasterio.open(roi_path) as src:
        roi = src.read(1)
        
    # Ensure shapes match
    if gt.shape != pred_binary.shape:
        raise ValueError(f"GT shape {gt.shape} != Pred shape {pred_binary.shape}")
        
    # Flatten arrays
    gt_flat = gt.flatten()
    pred_flat = pred_binary.flatten()
    roi_flat = roi.flatten()
    
    # Select only valid ROI pixels
    valid_indices = roi_flat > 0
    
    y_true = gt_flat[valid_indices]
    y_pred = pred_flat[valid_indices]
    
    # Binarize GT if needed (e.g. if 255 is water)
    y_true = (y_true > 0).astype(np.uint8)
    
    print(f"Evaluating on {len(y_true)} pixels within ROI...")
    
    # Calculate Metrics
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # IoU
    iou = tp / (tp + fp + fn + 1e-6)
    
    # Precision, Recall, F1, OA
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print("-" * 30)
    print(f"IoU:       {iou:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"OA:        {oa:.4f}")
    print(f"Kappa:     {kappa:.4f}")
    print("-" * 30)
    print(f"Confusion Matrix:\nTP={tp}, TN={tn}, FP={fp}, FN={fn}")

def main():
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup Model
    PATCH_SIZE = config['image_size']
    STRIDE = PATCH_SIZE // 2
    
    model = Model(in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model weights loaded.")
    else:
        print(f"Error: Weights not found at {MODEL_WEIGHTS_PATH}")
        return

    # Workflow
    if os.path.exists(INFERENCE_FILE_PATH):
        # Step 1: Inference
        prob_map, profile = predict_sliding_window(
            model=model,
            image_path=INFERENCE_FILE_PATH,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            num_classes=config['num_classes'],
            mean=NORM_MEAN,
            std=NORM_STD,
            device=device
        )
        
        # Step 2: Masking & Saving
        if os.path.exists(ROI_PATH):
            final_pred = apply_roi_mask_and_save(prob_map, ROI_PATH, RESULT_SAVE_PATH, profile)
            
            # Step 3: Evaluation
            if os.path.exists(GT_PATH):
                evaluate_metrics(final_pred, GT_PATH, ROI_PATH)
            else:
                print("GT_PATH not found. Skipping evaluation.")
        else:
            print("ROI_PATH not found. Skipping masking and evaluation.")
            # Save raw result if ROI missing
            # ... (Optional)
    else:
        print("Inference file not found.")

if __name__ == "__main__":
    main()