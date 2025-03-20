from model import UNET
from utils import (load_checkpoint)
import torch
from torchvision import transforms
from PIL import Image
import time
import os

def calculate_dice_score(predicted_mask, ground_truth_mask):

    intersection = (predicted_mask * ground_truth_mask).sum()
    dice_score = (2 * intersection) / (
        torch.sum(predicted_mask) + torch.sum(ground_truth_mask) + 1e-8
    )
    return dice_score.item()

def preprocess_image(image_path, input_size, device):
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # same as validation
    ])
    
    return transform(image).unsqueeze(0).to(device)

def load_ground_truth_mask(mask_path, input_size, device):
    mask_image = Image.open(mask_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.NEAREST), 
        transforms.ToTensor()
    ])
    return transform(mask_image).unsqueeze(0).to(device)

def evaluate_model(model, image_dir, mask_dir, image_filename, mask_filename, num_classes, device, scaling_factors):
    original_image_path = os.path.join(image_dir, image_filename)
    original_mask_path = os.path.join(mask_dir, mask_filename)

    original_image = Image.open(original_image_path)
    original_width, original_height = original_image.size
    original_size = (original_width, original_height)
    
    # --- Dummy Inference Run (Warm-up) ---
    print("Performing dummy inference run to warm-up...")
    dummy_input_size = original_size # Use original size for dummy run, or a representative size
    dummy_image_tensor = preprocess_image(original_image_path, dummy_input_size, device)
    with torch.no_grad():
        model(dummy_image_tensor) # Just run forward pass, no timing
    print("Warm-up complete.")
    # --- End Dummy Inference Run ---

    results = {}
    
    for scale_factor in scaling_factors:
        scaled_height = int(original_height * scale_factor)
        scaled_width = int(original_width * scale_factor)
        input_size = (scaled_height, scaled_width)
        
        print(f"\n--- Scaling Factor: {scale_factor:.2f} (Input Size: {input_size}) ---")
        
        image_tensor = preprocess_image(original_image_path, input_size, device)
        ground_truth_mask = load_ground_truth_mask(original_mask_path, input_size, device)
                
        start_time = time.time()
        with torch.no_grad():
            predicted_mask = torch.sigmoid(model(image_tensor))
        inference_time = time.time() - start_time
        
        predicted_mask_binary = (predicted_mask > 0.5).float()
        dice_score = calculate_dice_score(predicted_mask_binary, ground_truth_mask)
        results[scale_factor] = {
            "inference_time": inference_time,
            "dice_score": dice_score
        }
        
        print(f"Inference Time: {inference_time:.4f} seconds")
        print(f"Dice score: {dice_score:.4f}")
        
        
    print("\n--- Summary Results ---")
    for scale_factor, metrics in results.items():
        print(f"Scaling Factor: {scale_factor:.2f}, Inference Time: {metrics['inference_time']:.4f}s, Dice Score: {metrics['dice_score']:.4f}")
        



if __name__ == '__main__':
    model_path = './my_checkpoint.pth.tar' 
    image_dir = './train/'    
    mask_dir = './train_masks/'     
    image_filename = '0cdf5b5d0ce1_01.jpg'         
    mask_filename = '0cdf5b5d0ce1_01_mask.gif'
    num_classes = 1 
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA (GPU)")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU") 

    model = UNET(out_channels=num_classes)
    load_checkpoint(model_path, model, DEVICE)
    model.to(DEVICE)
    model.eval()
    
    scaling_factors = [2.0 , 1.5, 1.0, 0.75, 0.5, 0.25]
    
    evaluate_model(model, image_dir, mask_dir, image_filename, mask_filename, num_classes, DEVICE, scaling_factors)