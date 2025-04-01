from model import UNET
from utils import (load_checkpoint)
import torch
import torch.nn.functional as F # Import functional for interpolate
from torchvision import transforms
from PIL import Image
import time
import os

def calculate_dice_score(predicted_mask, ground_truth_mask):
    """Calculates the Dice score between two binary masks."""
    intersection = (predicted_mask * ground_truth_mask).sum()
    dice_score = (2 * intersection) / (
        torch.sum(predicted_mask) + torch.sum(ground_truth_mask) + 1e-8
    )
    return dice_score.item()

def preprocess_image(image_path, input_size, device):
    """Loads and preprocesses an image for model input."""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(input_size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) 
    ])
    
    return transform(image).unsqueeze(0).to(device)

def load_original_ground_truth_mask(mask_path, device):
    """Loads the ground truth mask at its original size."""
    mask_image = Image.open(mask_path).convert('L')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    original_size = mask_image.size # (width, height)
    original_hw = (original_size[1], original_size[0]) 
    return transform(mask_image).unsqueeze(0).to(device), original_hw

def evaluate_model(model, image_dir, mask_dir, image_filename, mask_filename, num_classes, device, scaling_factors):
    """
    Evaluates the model at different input scales.
    Inference time is measured on the scaled input.
    Dice score is calculated by resizing the predicted mask back to the original GT size.
    """
    original_image_path = os.path.join(image_dir, image_filename)
    original_mask_path = os.path.join(mask_dir, mask_filename)

    # --- Load Original Ground Truth Mask (ONCE) ---
    print("Loading original ground truth mask...")
    original_gt_mask_tensor, original_mask_hw = load_original_ground_truth_mask(original_mask_path, device)
    print(f"Original mask size (H, W): {original_mask_hw}")

    # Get original image size for scaling calculations
    original_image_pil = Image.open(original_image_path)
    original_width, original_height = original_image_pil.size


    # --- Dummy Inference Run (Warm-up) ---
    print("Performing dummy inference run to warm-up...")
    # Use a representative size, e.g., the largest scale factor or original size
    # Here using original size, but could use max scaled size too.
    dummy_input_size = (original_height, original_width)
    dummy_image_tensor = preprocess_image(original_image_path, dummy_input_size, device)
    with torch.no_grad():
        _ = model(dummy_image_tensor) # Just run forward pass, no timing or storage
    torch.cuda.synchronize() if device.type == 'cuda' else None # Ensure completion if on CUDA
    print("Warm-up complete.")
    # --- End Dummy Inference Run ---

    results = {}

    for scale_factor in scaling_factors:
        # Calculate scaled input size based on original image dimensions
        scaled_height = int(original_height * scale_factor)
        scaled_width = int(original_width * scale_factor)
        input_size = (scaled_height, scaled_width) # H, W for transforms.Resize

        print(f"\n--- Scaling Factor: {scale_factor:.2f} (Input Size: {input_size}) ---")

        # Preprocess the image to the current scaled size
        image_tensor = preprocess_image(original_image_path, input_size, device)

        # --- Inference ---
        start_time = time.time()
        with torch.no_grad():
            # Model outputs prediction at the scaled size
            predicted_mask_scaled = torch.sigmoid(model(image_tensor))
        torch.cuda.synchronize() if device.type == 'cuda' else None # Ensure completion for accurate timing
        inference_time = time.time() - start_time

        # --- Resize Predicted Mask to Original GT Size ---
        # predicted_mask_scaled shape: [1, C, H_scaled, W_scaled]
        # original_mask_hw: (H_original, W_original)
        print(f"Using technique bicubic")
        predicted_mask_original_size = F.interpolate(
            predicted_mask_scaled,
            size=original_mask_hw, # Target size (H, W)
            mode='bicubic',       
            # align_corners=False    # Recommended setting
        )

        # --- Threshold and Calculate Dice Score ---
        # Threshold the *resized* prediction
        predicted_mask_binary = (predicted_mask_original_size > 0.5).float()

        # Calculate Dice against the *original sized* ground truth mask
        dice_score = calculate_dice_score(predicted_mask_binary, original_gt_mask_tensor)

        # Store results
        results[scale_factor] = {
            "inference_time": inference_time,
            "dice_score": dice_score,
            "input_size": input_size
        }

        print(f"Input Size (H, W): {input_size}")
        print(f"Inference Time: {inference_time:.4f} seconds")
        print(f"Dice score (vs Original GT): {dice_score:.4f}")


    print("\n--- Summary Results ---")
    for scale_factor, metrics in results.items():
        print(f"Scaling Factor: {scale_factor:.2f}, Input Size: {metrics['input_size']}, Inference Time: {metrics['inference_time']:.4f}s, Dice Score: {metrics['dice_score']:.4f}")


if __name__ == '__main__':
    image_dir = './train/'
    mask_dir = './train_masks/'
    image_filename = '0cdf5b5d0ce1_01.jpg'
    mask_filename = '0cdf5b5d0ce1_01_mask.gif'
    num_classes = 1

    # --- Device Setup ---
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA (GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")

    # --- Model Loading ---
    print("Loading model...")
    model = UNET(in_channels=3, out_channels=num_classes)
    try:
        checkpoint_path = "my_checkpoint.pth.tar"
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        load_checkpoint(torch.load(checkpoint_path, map_location=DEVICE), model) 
        print(f"Checkpoint '{checkpoint_path}' loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the checkpoint file exists and the path is correct.")
        exit() # Exit if checkpoint is missing
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        exit()


    model.to(DEVICE)
    model.eval() 
    scaling_factors = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] 

    evaluate_model(model, image_dir, mask_dir, image_filename, mask_filename, num_classes, DEVICE, scaling_factors)