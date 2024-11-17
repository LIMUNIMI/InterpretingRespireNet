import numpy as np
import os

from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from posthoc import load_model, generate_aggregated_gradcam
from image_dataloader import image_loader

from tqdm import tqdm

#-------------------------------------------------------------------------------Load Dataset-------------------------------------------------------------------

def load_data(data_dir, folds_file):
    
    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                                params_json='params_json', input_transform=test_transform)
    return test_dataset

#-----------------------------------------------------------------------------Save and Load AGC----------------------------------------------------------------

def save_gradcam_data(gradcam_maps, labels, filename='gradcam_data.npz'):
    np.savez_compressed(filename, gradcam_maps=gradcam_maps, labels=labels)

def load_gradcam_data(filename='gradcam_data.npz'):
    data = np.load(filename)
    return data['gradcam_maps'], data['labels']

#-----------------------------------------------------------------------------------Main-----------------------------------------------------------------------

def main():
    
    gradcam_file = 'gradcam_data.npz'

    if os.path.exists(gradcam_file):
        print("Loading saved Aggregated Grad-CAM++...")
        agg_gradcams, labels = load_gradcam_data(gradcam_file)
    else:
        dataset = load_data('./data/icbhi_dataset/audio_text_data/', './data/patient_list_foldwise.txt')
        checkpoint_path = 'models/ckpt_best.pkl'
        model = load_model(checkpoint_path, device='cuda')

        agg_gradcams, labels = []  , []
        
        for image, label in tqdm(dataset, desc='Generating Aggregated Grad-CAM++...'):
            aggregated_gradcam_map = generate_aggregated_gradcam(model, image)
            agg_gradcams.append(aggregated_gradcam_map.flatten())
            labels.append(label)

        agg_gradcams = np.array(agg_gradcams)
        labels = np.array(labels)

        print('Saving generated Aggregated Grad-CAM++ maps...')
        save_gradcam_data(agg_gradcams, labels, gradcam_file)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(agg_gradcams, labels, test_size=0.3, random_state=42)

    # Searching best hyper-parameters
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    #     'kernel': ['rbf']
    # }

    # clf = svm.SVC()
    # grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=1)
    # grid_search.fit(X_train, y_train)
    # print(f'Best Hyper-Parameters SVM: {grid_search.best_params_}')
    # print(f'Best accuracy SVM: {grid_search.best_score_}')
          
    # best_svm_clf = grid_search.best_estimator_
              
    # Initialize the Stratified K-Fold cross-validation with 5 folds
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds)
    fold_accuracies = []  

    print(f'Training SVM with RBF kernel on {k_folds} folds:')
    
    # Train and evaluate the SVM model using K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        clf = svm.SVC(kernel='rbf', gamma='scale')
        clf.fit(X_fold_train, y_fold_train)

        # Evaluate the classifier on the validation set
        y_fold_pred = clf.predict(X_fold_val)
        fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
        fold_accuracies.append(fold_accuracy)

        
        print(f'Fold {fold + 1}/{k_folds} - Accuracy: {fold_accuracy:.4f}')

    mean_accuracy = np.mean(fold_accuracies)
    print(f'\nMean accuracy across {k_folds} folds: {mean_accuracy:.4f}')

    final_clf = svm.SVC(kernel='rbf', gamma='scale')
    final_clf.fit(X_train, y_train) 
    y_test_pred = final_clf.predict(X_test)  
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'\nTest accuracy with SVM (RBF kernel): {test_accuracy:.4f}')

if __name__ == '__main__':
    main()
