import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_sample_images(model, vis_loader, device, epoch, num_samples=12):
    model.eval()
    
    # Get one batch
    try:
        batch = next(iter(vis_loader))
    except StopIteration:
        return None

    images, labels, _ = batch
    
    # Select samples
    actual_num_samples = min(num_samples, images.shape[0])
    if actual_num_samples == 0:
        return None
        
    images = images[:actual_num_samples].to(device)
    labels = labels[:actual_num_samples] # Keep on CPU for plotting
    
    with torch.no_grad():
        outputs = model(images)
        # Handle deep supervision or multiple outputs
        if isinstance(outputs, (list, tuple)):
            main_output = outputs[0]
        else:
            main_output = outputs
            
        # Binary segmentation: sigmoid -> threshold
        pred_probs = torch.sigmoid(main_output)
        pred_seg = (pred_probs > 0.5).float()

    samples_data = []
    
    for i in range(actual_num_samples):
        # 1. Process Input Image (SAR VV band -> Grayscale)
        # Input is (2, H, W). VV is channel 0.
        img_tensor = images[i].cpu() # (2, H, W)
        vv = img_tensor[0]
        
        # Simple min-max normalization for visualization
        c_min, c_max = vv.min(), vv.max()
        if c_max - c_min > 1e-6:
            vv_norm = (vv - c_min) / (c_max - c_min)
        else:
            vv_norm = torch.zeros_like(vv)
            
        vv_img = vv_norm.numpy()
        
        # 2. Process Ground Truth
        label_img = labels[i].numpy() # (H, W) or (1, H, W)
        if label_img.ndim == 3:
            label_img = label_img[0]
            
        # 3. Process Prediction
        pred_img = pred_seg[i, 0].cpu().numpy() # (H, W)
        
        samples_data.append((vv_img, label_img, pred_img))

    num_rows = 3
    num_cols = actual_num_samples
    
    # Adjust figure size: wider to accommodate 12 columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(f'Epoch {epoch} - Model Predictions', fontsize=16, fontweight='bold')
    
    # Ensure axes is a 2D array for consistent indexing
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
        
    # Handle case where num_samples=1 and axes might be 1D array if not reshaped correctly above,
    # but subplots(3, 1) returns (3,) array.
    if axes.ndim == 1: 
        if num_cols == 1: axes = axes.reshape(-1, 1)
        else: axes = axes.reshape(1, -1)

    # --- Set Row Titles ---
    row_titles = ["Original Image (VV)", "Ground Truth", "Prediction"]
    for i in range(num_rows):
        axes[i, 0].set_ylabel(row_titles[i], fontsize=12, fontweight='bold')

    # --- Populate the Grid ---
    for i in range(num_cols): # Iterate through columns (samples)
        vv_img, label_img, pred_img = samples_data[i]
        
        # Column 1: Original Image (VV)
        axes[0, i].imshow(vv_img, cmap='gray')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Column 2: Ground Truth
        axes[1, i].imshow(label_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # Column 3: Segmentation Prediction
        axes[2, i].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
            
    plt.tight_layout(rect=[0.05, 0, 1, 0.95]) # Adjust layout
    
    return fig