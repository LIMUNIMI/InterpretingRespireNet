import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from scipy.spatial.distance import pdist
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load external modules
from image_dataloader import image_loader
from nets.network_cnn import model

# ------------------------------------------------------------------Loading Pre-Trained model---------------------------------------------------------------

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

# ---------------------------------------------------------------------Input_Perturbation------------------------------------------------------------------

def input_perturbation(model, image, device='cuda', batch_size=64):
    """Input perturbation"""
    
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


# --------------------------------------------------------------------Display_Importance_Map------------------------------------------------------------------

def display_importance_map(original_image, importance_map, predicted_class, original_prob, output_dir, sample_index, true_label):
    """Display importance map result from input perturbation"""
    
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
        importance_map = 2 * (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min()) - 1    
    else:
        importance_map = np.zeros_like(importance_map)  # Set to zeros if all values are the same

    # Create the subplot figure
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    fig.suptitle('Post-Hoc Explainability')

    # Original input image
    axs[0].imshow(original_image)
    axs[0].set_title(f'[Original Input]: {class_labels[predicted_class]} {"%.3f" % original_prob}')

    # Importance map with colorbar
    # Set the vmin and vmax to properly display the range of -1 to 1
    im = axs[1].imshow(importance_map, cmap='nipy_spectral', vmin=-1, vmax=1, alpha=0.8)
    axs[1].set_title('[Importance Map]')

    cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)  
    

    # Save the figure
    output_path = os.path.join(output_dir, f'sample_{sample_index}_importance_map.png')
    plt.savefig(output_path)
    plt.close()


# ----------------------------------------------------------------------Generate Grad-CAM-------------------------------------------------------------------

def generate_aggregated_gradcam(model, image, image_numpy, output_dir, sample_index, return_map=False):
    """Generate aggregated Grad-CAM++ visualization across multiple layers."""
    
    layers = [model.model_ft.layer2, model.model_ft.layer3, model.model_ft.layer4]  # Adjust based on architecture
    aggregated_cam = None

    for layer in layers:
        cam = GradCAMPlusPlus(model=model, target_layers=[layer])
        grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0]
        aggregated_cam = grayscale_cam if aggregated_cam is None else aggregated_cam + grayscale_cam

    # Average the Grad-CAM maps from all layers
    aggregated_cam /= len(layers)
    visualization = show_cam_on_image(image_numpy, aggregated_cam, use_rgb=True)

    # Create figure and save visualization
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    fig.suptitle('Post-Hoc Explainability - Aggregated Grad-CAM++')
    axs[0].imshow(image_numpy)
    axs[0].set_title('[Original Input]')
    im = axs[1].imshow(visualization)
    axs[1].set_title(f'[Aggregated Grad-CAM++]')
    fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)

    output_path = os.path.join(output_dir, f'sample_{sample_index}_aggregated_gradcampp.png')
    plt.savefig(output_path)
    plt.close()

    if return_map:
        return aggregated_cam


# ----------------------------------------------------------------------Generate smoothGrad------------------------------------------------------------------

def smoothgrad_camplusplus(model, image, image_numpy, output_dir, sample_index, n_samples=25, noise_level=0.1, return_map=False):
    """Generate SmoothGrad-CAM++ visualization by averaging Grad-CAM++ results over noisy input copies."""
    
    target_layer = model.model_ft.layer4
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    # Accumulate Grad-CAM++ maps across multiple noisy samples
    smooth_gradcam = np.zeros_like(image_numpy[..., 0])
    for _ in range(n_samples):
        noise = noise_level * torch.randn_like(image).to(image.device)
        noisy_image = image + noise
        grayscale_cam = cam(input_tensor=noisy_image.unsqueeze(0))[0]
        smooth_gradcam += grayscale_cam

    # Average and normalize the result
    smooth_gradcam /= n_samples
    visualization = show_cam_on_image(image_numpy, smooth_gradcam, use_rgb=True)

    # Create figure and save visualization
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    fig.suptitle('Post-Hoc Explainability - SmoothGrad-CAM++')
    axs[0].imshow(image_numpy)
    axs[0].set_title('[Original Input]')
    im = axs[1].imshow(visualization)
    axs[1].set_title('[SmoothGrad-CAM++]')
    fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)

    output_path = os.path.join(output_dir, f'sample_{sample_index}_smoothgrad_campp.png')
    plt.savefig(output_path)
    plt.close()

    if return_map:
        return smooth_gradcam


# ---------------------------------------------------------------------Mean and Variance L1-----------------------------------------------------------------

def importance_map_stability(model, image, device='cuda'):
    """Compute stability of the importance map."""
    
    X = [input_perturbation(model, image, device)[0].flatten() for _ in range(6)]
    
    # L1 distance matrix 
    L1 = pdist(X, metric='minkowski', p=1)
    
    # Compute mean and variance of L1 
    mean_L1 = np.mean(L1)
    var_L1 = np.var(L1)
    
    return mean_L1, var_L1


# -----------------------------------------------------------------------------Main-------------------------------------------------------------------------

def main(data_dir, checkpoint, folds_file, output_dir, sample_index):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize data loader
    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                                params_json='params_json', input_transform=test_transform)

    # Load model
    model = load_model(checkpoint, device)
    if model is not None:
        print('Model loaded successfully.')
    else: 
        print('Error: Model failed to load. Please check the checkpoint path and device.')
        raise ValueError('Model loading failed')
        
    for index in sample_index:
        print('Post-Hoc Explainability...sample index: ', index)
        image, label = test_dataset[index]

        # Generate explanation for the image
        importance_map, predicted_class, original_prob, all_probs = input_perturbation(model, image, device)
        
        """Uncomment this part to compute mean and variance of L1 matrix"""
        # mean_L1, var_L1 = importance_map_stability(model, image, device)
        
        # print(f'Mean of the matrix of distances L1:', mean_L1)
        # print(f'Variance of the matrix of distances L1:',var_L1)
        
        """"""
        
        # Convert the image to numpy for visualization
        mean = np.array([0.5091, 0.1739, 0.4363])
        std = np.array([0.2288, 0.1285, 0.0743])
        image_numpy = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        image_numpy = (image_numpy * std) + mean  # Denormalize
        image_numpy = np.clip(image_numpy, 0, 1)

        # Display importance map
        display_importance_map(image_numpy, importance_map, predicted_class, original_prob, output_dir, index, label)
        
        # Generate Grad-CAM++
        aggregated_gradcam_map = generate_aggregated_gradcam(model, image, image_numpy, output_dir, index, return_map=True)
        smooth_gradcam_map = smoothgrad_camplusplus(model, image, image_numpy, output_dir, sample_index, n_samples=25, noise_level=0.1, return_map=True)

        # Print out the true label and predicted label
        print(f'True Label: {class_labels[label]}')
        print(f'Predicted Label: {class_labels[predicted_class]} with probability {'%.3f' % original_prob}')
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='RespireNet: Post-Hoc Explanation of Lung Sound Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--folds_file', type=str, required=True, help='Path to the folds file for dataset split')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the outputs')
    parser.add_argument('--sample_index', type=int, nargs='+', required=True, help='List of sample indices for explanation')

    args = parser.parse_args()
    main(args.data_dir, args.checkpoint, args.folds_file, args.output_dir, args.sample_index)
