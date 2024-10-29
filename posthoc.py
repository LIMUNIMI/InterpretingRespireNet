import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor
import os
import argparse
import random

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from image_dataloader import image_loader
from nets.network_cnn import model

#------------------------------------------------------------------Loading Pre-Trained model---------------------------------------------------------------
# Load pre-trained model
def load_model(checkpoint_path, device='cuda'):
    net = model(num_classes=4).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    net.load_state_dict(checkpoint)
    net.eval()
    return net

# Class labels
class_labels = {
    0: 'Normal',
    1: 'Crackle',
    2: 'Wheeze',
    3: 'Both Crackle and Wheeze'
}

#---------------------------------------------------------------------Input_Perturbation------------------------------------------------------------------

def input_perturbation(model, image, device='cuda', batch_size=64):
    
    # Forwarding the original input into the model obtaining the 'original prob'
    image = image.to(device)
    image = image.unsqueeze(0)  # Add batch dimension
    original_output = model(image)
    original_probs = torch.nn.functional.softmax(original_output, dim=1)
    _, original_pred = torch.max(original_output, 1)
    original_pred_class = original_pred.item()
    original_prob = original_probs[0, original_pred_class].item()

    # Initialize an importance map
    importance_map = torch.zeros(image.shape[-2], image.shape[-1]).to(device)
    
    # Get image dimensions
    _, _, H, W = image.shape
    
    # Number of perturbations (1 per pixel)
    total_pixels = H * W

    # Process perturbations in batches
    for batch_start in tqdm(range(0, total_pixels, batch_size)):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_size_current = batch_end - batch_start

        # Random indices perturbation (o to B Random)
        random_indices = torch.randperm(batch_size_current)
        perturbed_images = image.repeat(batch_size_current, 1, 1, 1)

        for i in range(batch_size_current):
            idx = batch_start + random_indices[i].item()
            row = idx // W
            col = idx % W
            perturbed_images[i, :, row, col] = 0  

        with torch.no_grad():
            perturbed_output = model(perturbed_images)
            perturbed_probs = torch.nn.functional.softmax(perturbed_output, dim=1)
            perturbed_prob = perturbed_probs[:, original_pred_class]

        importance_values = original_prob - perturbed_prob

        for i in range(batch_size_current):
            idx = batch_start + random_indices[i].item()
            row = idx // W
            col = idx % W
            importance_map[row, col] = importance_values[i]
            
    return importance_map.cpu(), original_pred_class, original_prob, original_probs

#------------------------------------------------------------------------Display_Results------------------------------------------------------------------

def display_importance_map(original_image, importance_map, predicted_class, original_prob, output_dir, sample_index, true_label):
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert to numpy if necessary
    if isinstance(importance_map, torch.Tensor):
        importance_map = importance_map.numpy()

    # Clip the original image to valid range for display (0-1 for floats)
    original_image = np.clip(original_image, 0, 1)

    # Normalize importance map
    if importance_map.max() != importance_map.min():
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())
    else:
        importance_map = np.zeros_like(importance_map)  # Set to zeros if all values are the same

    # Plot and save the image with the importance map
    plt.figure(figsize=(16, 8), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"[Original Input] \nPredicted Class: {class_labels[predicted_class]} {'%.3f' % original_prob}")

    plt.subplot(1, 2, 2)
    plt.imshow(original_image, alpha=0.4)
    plt.imshow(importance_map, cmap='nipy_spectral', alpha=0.8)  # Show the importance map in heatmap format
    plt.title(f"[Importance Map]")
    
    plt.colorbar()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'sample_{sample_index}_importance_map.png')
    plt.savefig(output_path)
    plt.close()


# ----------------------------------------------------------------------Generate Grad-CAM-------------------------------------------------------------------

def generate_gradcam(model, image, predicted_class, image_numpy, output_dir, sample_index):
    """Generate and save GradCAM visualization."""
    target_layer = model.model_ft.layer4
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0]

    # Overlay GradCAM on original image
    visualization = show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(12, 6), dpi=300)
    # original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(image_numpy)
    plt.title("[Original Input]")

    # GradCAM on the right
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"[GradCAM]\nPredicted Class: {class_labels[predicted_class]}")

    output_path = os.path.join(output_dir, f'sample_{sample_index}_gradcam.png')
    plt.savefig(output_path)
    plt.close()
    
    
#-----------------------------------------------------------------------------Main--------------------------------------------------------------------------

def main(data_dir, checkpoint, folds_file, output_dir, sample_index):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize data loader
    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                                params_json="params_json", input_transform=test_transform)

    # Load model
    model = load_model(checkpoint, device)
    if model is not None:
        print("Model loaded successfully.")
    else: 
        print("Error: Model failed to load. Please check the checkpoint path and device.")
        raise ValueError("Model loading failed")
        
    for index in sample_index:
        print('Post-Hoc Explainability...sample index: ', index)
        image, label = test_dataset[index]

        # Generate explanation for the image
        importance_map, predicted_class, original_prob, all_probs = input_perturbation(model, image, device)

        # Convert the image to numpy for visualization
        # Denormalize the image for visualization
        mean = np.array([0.5091, 0.1739, 0.4363])
        std = np.array([0.2288, 0.1285, 0.0743])
        image_numpy = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        image_numpy = (image_numpy * std) + mean  # Denormalize
        image_numpy = np.clip(image_numpy, 0, 1)

        # Save the importance map and results
        display_importance_map(image_numpy, importance_map, predicted_class, original_prob, output_dir, index, label)

        # GradCAM
        generate_gradcam(model, image, predicted_class, image_numpy, output_dir, index)
        
        # Print out the true label and predicted label
        print(f"True Label: {class_labels[label]}")
        print(f"Predicted Label: {class_labels[predicted_class]} with probability {'%.3f' % original_prob}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RespireNet: Post-Hoc Explanation of Lung Sound Classification')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')
    parser.add_argument('--folds_file', type=str, help='patient list foldwise')
    parser.add_argument('--output_dir', type=str, help='xai results saving directory')
    parser.add_argument('--sample_index', type=int, nargs='+', help='sample indices as list of integers')
    
    args = parser.parse_args()
    main(args.data_dir, args.checkpoint, args.folds_file, args.output_dir, args.sample_index)
