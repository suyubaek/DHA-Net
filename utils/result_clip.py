
import os
import numpy as np
import rasterio
from PIL import Image

def clip_result_with_mask(pred_path, mask_path, save_path):
    """
    Clips the prediction image using the ROI mask.
    Pixels outside the ROI (mask != 1) will be transparent.
    """
    print(f"Processing: {pred_path}")
    
    # Check if files exist
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} does not exist.")
        return
    if not os.path.exists(mask_path):
        print(f"Error: {mask_path} does not exist.")
        return

    # Read Prediction TIF
    # Using rasterio for TIF handling to preserve georeference if needed (though we save as PNG here)
    # Or just cv2/PIL for simplicity if strictly for visualization
    
    with rasterio.open(pred_path) as src_pred:
        pred_data = src_pred.read(1) # Read first band
        profile = src_pred.profile

    # Read ROI Mask TIF
    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        
    # Ensure shapes match
    if pred_data.shape != mask_data.shape:
        print(f"Shape Mismatch: Pred {pred_data.shape} vs Mask {mask_data.shape}")
        # Resize mask to match pred using PIL
        mask_img = Image.fromarray(mask_data)
        mask_img = mask_img.resize((pred_data.shape[1], pred_data.shape[0]), resample=Image.NEAREST)
        mask_data = np.array(mask_img)

    # Create destination profile
    profile.update({
        'driver': 'GTiff',
        'height': pred_data.shape[0],
        'width': pred_data.shape[1],
        'alpha': 'no' # We will use nodata for transparency typically, or could use 'yes' for alpha band
    })
    
    # Determine NoData value based on dtype to support transparency
    # If float, use NaN. If int, pick a value outside range (e.g. 255 for uint8 if max is 1)
    dtype = pred_data.dtype
    nodata_val = None
    
    if np.issubdtype(dtype, np.floating):
        nodata_val = np.nan
    elif dtype == np.uint8:
        # Assuming binary 0/1 or similar small ints. 255 is safe nodata.
        nodata_val = 255
    else:
        # detailed handling, defaulting to a common nodata like -9999
        nodata_val = -9999
        
    profile.update(nodata=nodata_val)
    
    # Apply Mask: Set outside ROI to nodata_val
    # mask_data is 1 for valid, others invalid
    # Ensure mask is binary logical
    valid_mask = (mask_data == 1)
    
    # Prepare output data
    out_data = pred_data.copy()
    
    # Fill Nodata
    if nodata_val is not None:
         # Need to ensure out_data can hold nodata_val if it's outside original range?
         # If uint8 (0-255), 255 is fine.
         # If int8, -9999 bad.
         out_data[~valid_mask] = nodata_val
         
    # Save as TIF
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(out_data, 1)
        
    print(f"Saved clipped TIF result to: {save_path}")

if __name__ == "__main__":
    base_dir = "/Users/song/Projects/lancing"
    pred_file = os.path.join(base_dir, "lancang_infer/lancang_river_Attentive_UNet.tif")
    mask_file = os.path.join(base_dir, "lancang_infer/roi_mask.tif")
    # Save as _clip.tif
    save_file = pred_file.replace(".tif", "_clip.tif")
    
    try:
        clip_result_with_mask(pred_file, mask_file, save_file)
    except Exception as e:
        print(f"An error occurred: {e}")
