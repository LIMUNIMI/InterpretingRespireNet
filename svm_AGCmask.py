import numpy as np
import os

from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from posthoc_backup import load_model, generate_aggregated_gradcam
from image_dataloader import image_loader

from tqdm import tqdm

#-------------------------------------------------------------Dataset Load-------------------------------------------------------------------

def load_data(data_dir, folds_file):
    # Define the transformations for input images
    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    # Load the dataset using the custom image loader
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                                params_json='params_json', input_transform=test_transform)
    return test_dataset

#-----------------------------------------------------Save and Load Agg_Gradcam_Mask----------------------------------------------------------

def save_gradcam_data(gradcam_maps, labels, filename='gradcam_data.npz'):
    np.savez_compressed(filename, gradcam_maps=gradcam_maps, labels=labels)


def load_gradcam_data(filename='gradcam_data.npz'):
    data = np.load(filename)
    return data['gradcam_maps'], data['labels']

#---------------------------------------------------------------Main--------------------------------------------------------------------------

def main():
    gradcam_file = 'gradcam_data.npz'

    # Check if the Grad-CAM maps have already been saved to avoid recomputing them
    if os.path.exists(gradcam_file):
        print("Loading saved Grad-CAM maps...")
        agg_gradcams, labels = load_gradcam_data(gradcam_file)
    else:
        # Load dataset and model if the Grad-CAM maps are not saved
        dataset = load_data('./data/icbhi_dataset/audio_text_data/', './data/patient_list_foldwise.txt')
        checkpoint_path = 'models/ckpt_best.pkl'
        model = load_model(checkpoint_path, device='cuda')

        # Initialize lists to store Grad-CAM maps and their corresponding labels
        agg_gradcams = []  
        labels = []        

        # Generate maps for each image in the dataset
        for image, label in tqdm(dataset, desc='Generating Grad-CAMs'):
            aggregated_gradcam_map = generate_aggregated_gradcam(model, image)
            agg_gradcams.append(aggregated_gradcam_map.flatten())
            labels.append(label)

        # Convert  maps and labels into numpy arrays for saving
        agg_gradcams = np.array(agg_gradcams)
        labels = np.array(labels)

        # Save the generated  maps and labels to a file
        print("Saving generated Grad-CAM maps...")
        save_gradcam_data(agg_gradcams, labels, gradcam_file)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(agg_gradcams, labels, test_size=0.3, random_state=42)

    # Initialize the Stratified K-Fold cross-validation with 5 folds
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds)
    fold_accuracies = []  

    print(f"Training SVM with RBF kernel on {k_folds} folds:")
    
    # Train and evaluate the SVM model using K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Create an SVM classifier with RBF kernel and train it on the current fold
        clf = svm.SVC(kernel='rbf', gamma='scale')
        clf.fit(X_fold_train, y_fold_train)

        # Evaluate the classifier on the validation set
        y_fold_pred = clf.predict(X_fold_val)
        fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
        fold_accuracies.append(fold_accuracy)

        
        print(f'Fold {fold + 1}/{k_folds} - Accuracy: {fold_accuracy:.4f}')

    # Calculate the average accuracy across all folds
    mean_accuracy = np.mean(fold_accuracies)
    print(f'\nMean accuracy across {k_folds} folds: {mean_accuracy:.4f}')

    # Final evaluation of the SVM classifier on the test set
    final_clf = svm.SVC(kernel='rbf', gamma='scale')
    final_clf.fit(X_train, y_train) 
    y_test_pred = final_clf.predict(X_test)  
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'\nTest accuracy with SVM (RBF kernel): {test_accuracy:.4f}')

if __name__ == '__main__':
    main()
