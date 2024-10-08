import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor
import cv2
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import external module
from image_dataloader import image_loader
from nets.network_cnn import model
from utils import create_mel_raw

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

# -------------------------------------------------------------------patch_perturbation-------------------------------------------------------------------------

def patch_perturbation(model, image, device='cuda', patch_size=10):
    
    image = image.to(device)
    image = image.unsqueeze(0)  # Add batch dimension
    original_output = model(image)
    original_probs = torch.nn.functional.softmax(original_output, dim=1)
    _, original_pred = torch.max(original_output, 1)
    original_pred_class = original_pred.item()
    original_prob = original_probs[0, original_pred_class].item()

    # Initialize an importance map and a list to store probability changes
    importance_map = torch.zeros(image.shape[-2], image.shape[-1])
    class_prob_differences = []

    # Get image dimensions
    _, _, H, W = image.shape

    # Generate perturbed images by masking 
    for i in tqdm(range(0, H, patch_size)):
        for j in range(0, W, patch_size): 
            perturbed_image = image.clone()
            # Mask out the patch
            perturbed_image[:, :, i:i+patch_size, j:j+patch_size] = 0
            # Forward pass on perturbed image
            perturbed_output = model(perturbed_image)
            perturbed_probs = torch.nn.functional.softmax(perturbed_output, dim=1)
            perturbed_prob = perturbed_probs[0, original_pred_class].item()
            # Calculate importance as the drop in probability
            importance = original_prob - perturbed_prob
            importance = max(importance, 0)  # Ensure non-negative
            # Assign importance to the patch region
            importance_map[i:i+patch_size, j:j+patch_size] = importance
            # Append probability changes for all classes
            class_prob_differences.append(perturbed_probs.cpu().detach().numpy()[0])

    return importance_map.cpu(), original_pred_class, original_prob, original_probs, class_prob_differences

# -------------------------------------------------------------------display_importance_map-------------------------------------------------------------------------

def display_importance_map(original_image, importance_map, predicted_class, original_prob, all_probs, output_dir, sample_index, class_prob_differences, true_label):
    
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
    plt.title(f"Mel Spectrogram\n[Predicted class]: {class_labels[predicted_class]} {"%.3f" % original_prob}")

    plt.subplot(1, 2, 2)
    plt.imshow(original_image, alpha=0.6)
    plt.imshow(importance_map, cmap='jet', alpha=0.4)  # Show the importance map in heatmap format
    plt.title("Importance Map of the Mel Spectrogram")
    
    # Save the figure
    output_path = os.path.join(output_dir, f'sample_{sample_index}_importance_map.png')
    plt.savefig(output_path)
    plt.close()

#-----------------------------------------------------------------------------main--------------------------------------------------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_dir = './data/icbhi_dataset/audio_text_data/'
    checkpoint_path = 'models/ckpt_best.pkl'
    folds_file = './data/patient_list_foldwise.txt'
    output_dir = './xai_results/'  

    # Initialize data loader
    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                                params_json="params_json", input_transform=test_transform)

    # Load model
    model = load_model(checkpoint_path, device)

    # Get a test sample 
    sample_index = 10
    image, label = test_dataset[sample_index]

    # Generate explanation for the image
    importance_map, predicted_class, original_prob, all_probs, class_prob_differences = patch_perturbation(model, image, device, patch_size=10)

    # Convert the image to numpy for visualization
    # Denormalize the image for visualization
    mean = np.array([0.5091, 0.1739, 0.4363])
    std = np.array([0.2288, 0.1285, 0.0743])
    image_numpy = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    image_numpy = (image_numpy * std) + mean  # Denormalize
    image_numpy = np.clip(image_numpy, 0, 1)

    # Save the importance map and results
    display_importance_map(image_numpy, importance_map, predicted_class, original_prob, all_probs, output_dir, sample_index, class_prob_differences, label)

    # Print out the true label and predicted label
    print(f"True Label: {class_labels[label]}")
    print(f"Predicted Label: {class_labels[predicted_class]} with probability {original_prob*100:.2f}%")

if __name__ == "__main__":
    main()