import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random

from scipy.spatial.distance import pdist
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load external modules
from image_dataloader import image_loader
from nets.network_cnn import model

# ------------------------------------------------------------------Loading Pre-Trained model------------------------------------------------------------------

# Load pre-trained model from nets.network_cnn
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

# ---------------------------------------------------------------------Input_Perturbation----------------------------------------------------------------------

def input_perturbation(model, image, device='cuda', batch_size=64):
    # Forwarding input_data into model 
    image = image.to(device).unsqueeze(0)
    original_output = model(image)
    original_probs = F.softmax(original_output, dim=1)
    _, original_pred = torch.max(original_output, 1)
    original_pred_class = original_pred.item()
    original_prob = original_probs[0, original_pred_class].item()

    # Initialize and importance map with the same dimensions as the input
    importance_map = torch.zeros(image.shape[-2], image.shape[-1]).to(device)
    
    # Extract spacial dimensions from the image tensor
    _, _, H, W = image.shape
    pixel_size = 4
    
    # Define the total number of the perturbation required 
    total_pixels = (H // pixel_size) * (W // pixel_size)

    # Perturbation process developed in batches
    for batch_start in tqdm(range(0, total_pixels, batch_size)):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_size_current = batch_end - batch_start

        # Random indices perturbation 
        random_indices = torch.randperm(batch_size_current)
        perturbed_images = image.repeat(batch_size_current, 1, 1, 1)

        # Zero out (Perturbation of pixels)
        for i in range(batch_size_current):
            idx = batch_start + random_indices[i].item()
            row = (idx // (W // pixel_size)) * pixel_size
            col = (idx % (W // pixel_size)) * pixel_size
            perturbed_images[i, :, row:row+pixel_size, col:col+pixel_size] = 0  
        
        # Forwarding input_data perturbed into model 
        with torch.no_grad():
            perturbed_output = model(perturbed_images)
            perturbed_probs = F.softmax(perturbed_output, dim=1)
            perturbed_prob = perturbed_probs[:, original_pred_class]

        # Calculate the drop in probability between y (original prob) and y' (perturbed_prob)
        importance_values = original_prob - perturbed_prob

        # Update the importance_map
        for i in range(batch_size_current):
            idx = batch_start + random_indices[i].item()
            row = (idx // (W // pixel_size)) * pixel_size
            col = (idx % (W // pixel_size)) * pixel_size
            importance_map[row:row+pixel_size, col:col+pixel_size] = importance_values[i]
            
    return importance_map.cpu(), original_pred_class, original_prob, original_probs

# --------------------------------------------------------------------Display_Importance_Map-------------------------------------------------------------------

def display_importance_map(original_image, importance_map, predicted_class, original_prob, output_dir, sample_index, true_label):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(importance_map, torch.Tensor):
        importance_map = importance_map.numpy()

    original_image = np.clip(original_image, 0, 1)

    # Normalize the importance_map between values -1 and 1 as we need to know negative and positive influences
    if importance_map.max() != importance_map.min():
        importance_map = 2 * (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min()) - 1    
    else:
        importance_map = np.zeros_like(importance_map)
        
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    fig.suptitle('Post-Hoc Explainability')

    # Display the original_image with Predicted Class label 
    axs[0].imshow(original_image)
    axs[0].set_title(f'[Original Input]: Predicted Class:{class_labels[predicted_class]} {"%.3f" % original_prob}')

    # Importance map with colorbar
    # Set the vmin and vmax to properly display the range of -1 to 1
    im = axs[1].imshow(importance_map, cmap='turbo', vmin=-1, vmax=1, alpha=0.8)
    axs[1].set_title('[Importance Map]')

    cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)  
    
    output_path = os.path.join(output_dir, f'sample_{sample_index}_importance_map.png')
    plt.savefig(output_path)
    plt.close()

# ---------------------------------------------------------------------Generate Aggregated Grad-CAM------------------------------------------------------------

def generate_aggregated_gradcam(model, image):
    # Generate aggregated Grad-CAM++ map across multiple layers
    layers = [model.model_ft.layer2, model.model_ft.layer3, model.model_ft.layer4]  
    aggregated_cam = None

    for layer in layers:
        cam = GradCAMPlusPlus(model=model, target_layers=[layer])
        grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0]
        aggregated_cam = grayscale_cam if aggregated_cam is None else aggregated_cam + grayscale_cam

    # Average the Grad-CAM maps from all layers
    aggregated_cam /= len(layers)

    return aggregated_cam

# -----------------------------------------------------------------------Generate SmoothGrad-CAM++-------------------------------------------------------------

def smoothgrad_camplusplus(model, image, n_samples=25, noise_level=0.1):
    #Generate SmoothGrad-CAM++ map by averaging Grad-CAM++ results over noisy input copies
    target_layer = model.model_ft.layer4
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    original_height, original_width = image.shape[1], image.shape[2]

    # Accumulate Grad-CAM++ maps across multiple noisy samples
    smooth_gradcam = np.zeros((original_height, original_width))

    for _ in range(n_samples):
        noise = noise_level * torch.randn_like(image).to(image.device)
        noisy_image = image + noise
        grayscale_cam = cam(input_tensor=noisy_image.unsqueeze(0))[0]
        # Resize the grayscale CAM to match the original image dimensions
        grayscale_cam_resized = F.interpolate(torch.tensor(grayscale_cam).unsqueeze(0).unsqueeze(0), 
                                              size=(original_height, original_width), 
                                              mode='bilinear', align_corners=False).squeeze().numpy()
        smooth_gradcam += grayscale_cam_resized
    smooth_gradcam /= n_samples

    return smooth_gradcam


# ----------------------------------------------------------------------Visualize Aggregated Grad-CAM----------------------------------------------------------

def visualize_gradcam(image_numpy, cam_map, output_dir, sample_index, visualization_type='Aggregated Grad-CAM++'):
    
    visualization = show_cam_on_image(image_numpy, cam_map, use_rgb=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    fig.suptitle(f'Post-Hoc Explainability - {visualization_type}')
    axs[0].imshow(image_numpy)
    axs[0].set_title('[Original Input]')
    im = axs[1].imshow(visualization, vmin=0, vmax=1)
    axs[1].set_title(f'[{visualization_type}]')
    
    cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    
    output_path = os.path.join(output_dir, f'sample_{sample_index}_{visualization_type.replace(" ", "_").lower()}.png')
    plt.savefig(output_path)
    plt.close()

# -----------------------------------------------------------------------Visualize SmoothGrad-CAM++------------------------------------------------------------

def visualize_smoothgrad_camplusplus(image_numpy, smooth_gradcam_map, output_dir, sample_index, visualization_type='SmoothGrad-CAM++'):

    visualization = show_cam_on_image(image_numpy, smooth_gradcam_map, use_rgb=True)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    
    fig.suptitle('Post-Hoc Explainability - SmoothGrad-CAM++')
    axs[0].imshow(image_numpy)
    axs[0].set_title('[Original Input]')
    im = axs[1].imshow(visualization, vmin=0, vmax=1)
    axs[1].set_title(f'[{visualization_type}]')
    
    cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    
    output_path = os.path.join(output_dir, f'sample_{sample_index}_{visualization_type}.png')
    plt.savefig(output_path)
    plt.close()

# -------------------------------------------------------------------------Mean and Variance L1----------------------------------------------------------------

def importance_map_stability(model, image, device='cuda'):
    # Compute the stability of the input_perturbation (importance_map)
    torch.manual_seed(42)
    X = [input_perturbation(model, image, device)[0].flatten() for seed in range(4)]
    
    # L1 distance matrix 
    L1 = pdist(X, metric='minkowski', p=1)
    
    # Compute mean and variance of L1 
    mean_L1 = np.mean(L1)
    var_L1 = np.var(L1)
    
    return mean_L1, var_L1

# --------------------------------------------------------------------------------Main-------------------------------------------------------------------------

def main(data_dir, checkpoint, folds_file, output_dir, sample_index):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the data loader
    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                                params_json='params_json', input_transform=test_transform)

    # Load model
    model = load_model(checkpoint, device)
    if model is not None:
        print('Model loaded successfully.')
    else: 
        print('Error: Model failed to load. Please check the checkpoint_path and device.')
        raise ValueError('Model loading failed.')
        
    for index in sample_index:
        # Post-Hoc explainability process based on sample index for the future implementation
        print('Post-Hoc Explainability...sample_index: ', index)
        image, label = test_dataset[index]

        # Generate input_perturbation
        importance_map, predicted_class, original_prob, all_probs = input_perturbation(model, image, device)
        
        # Convert the image to numpy for visualization
        mean = np.array([0.5091, 0.1739, 0.4363])
        std = np.array([0.2288, 0.1285, 0.0743])
        # Convert from (C, H, W) to (H, W, C)
        image_numpy = image.permute(1, 2, 0).cpu().numpy() 
        image_numpy = (image_numpy * std) + mean  
        image_numpy = np.clip(image_numpy, 0, 1)

        # Display importance map
        display_importance_map(image_numpy, importance_map, predicted_class, original_prob, output_dir, index, label)
        
        # Generate and visualize Aggregated Grad-CAM++
        aggregated_gradcam_map = generate_aggregated_gradcam(model, image)
        visualize_gradcam(image_numpy, aggregated_gradcam_map, output_dir, index, 'Aggregated Grad-CAM++')

        # Generate and visualize SmoothGrad-CAM++
        smooth_gradcam_map = smoothgrad_camplusplus(model, image)
        visualize_smoothgrad_camplusplus(image_numpy, smooth_gradcam_map, output_dir, index)

        # Print out the true label and predicted label
        print(f'True Label: {class_labels[label]}')
        print(f'Predicted Label: {class_labels[predicted_class]} with probability {'%.3f' % original_prob}')
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='RespireNet: Post-Hoc Explanation of Lung Sound Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--folds_file', type=str, required=True, help='Path to the folds file for dataset split')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the outputs')
    parser.add_argument('--sample_index', type=int, nargs='+', required=True, help='List of sample indices for explanation or single sample index or entire dataset')

    args = parser.parse_args()
    main(args.data_dir, args.checkpoint, args.folds_file, args.output_dir, args.sample_index)
