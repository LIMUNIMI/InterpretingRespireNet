import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from image_dataloader import image_loader
from posthoc import load_model, input_perturbation

#-----------------------------------------------------------------------------Load Dataset---------------------------------------------------------------------

def load_data(data_dir, folds_file):

    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                        params_json='params_json', input_transform=test_transform)
    return test_dataset

#---------------------------------------------------------------------Save and Load Importance_Maps------------------------------------------------------------

def save_importance_data(importance_maps, labels, filename='importance_maps.npz'):
    np.savez_compressed(filename, importance_maps=importance_maps, labels=labels)

def load_importance_data(filename='importance_maps.npz'):
    data = np.load(filename)
    return data['importance_maps'], data['labels']

#--------------------------------------------------------------------------------Main--------------------------------------------------------------------------

def main():
    
    importance_map_file = 'importance_maps.npz'
    if os.path.exists(importance_map_file):
        print("Loading saved Importance_Maps...")
        importance_maps, labels = load_importance_data()
    else:
        dataset = load_data('./data/icbhi_dataset/audio_text_data/', './data/patient_list_foldwise.txt')
        checkpoint_path = 'models/ckpt_best.pkl'
        model = load_model(checkpoint_path, device='cuda')

        importance_maps, labels = [], []

        for image, label in tqdm(dataset, desc='Generating Importance_Maps...'):
            importance_map, _, _, _ = input_perturbation(model, image)
            importance_maps.append(importance_map.flatten())
            labels.append(label)

        importance_maps = np.array(importance_maps)
        labels = np.array(labels)

        # Salva le mappe generate
        print('Saving generated Importance_maps...')
        save_importance_data(importance_maps, labels, importance_map_file)

    # Suddivisione in set di training e testing
    X_train, X_test, y_train, y_test = train_test_split(importance_maps, labels, test_size=0.3, random_state=42)

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
