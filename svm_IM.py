import json
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv  # Enabling HalvingGridSearchCV
from sklearn import svm
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.model_selection import StratifiedKFold

from posthoc import load_model, input_perturbation
from image_dataloader import image_loader

from tqdm import tqdm

OUTPUT_DIR = 'svm_im'

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

BEST_PARAMS_FILE = os.path.join(OUTPUT_DIR, 'best_svm_im_params.json')

#-----------------------------------------------------------------------------Load Dataset---------------------------------------------------------------------

def load_data(data_dir, folds_file):

    test_transform = Compose([ToTensor(), Normalize(mean=[0.5091, 0.1739, 0.4363],
                                                    std=[0.2288, 0.1285, 0.0743])])
    test_dataset = image_loader(data_dir, folds_file, test_fold=4, train_flag=False,
                        params_json='params_json', input_transform=test_transform)
    return test_dataset

#---------------------------------------------------------------------Save and Load Importance_Maps------------------------------------------------------------

def save_importance_data(importance_maps, labels, filename=os.path.join(OUTPUT_DIR, 'importance_maps.npz')):
    ensure_output_dir()
    np.savez_compressed(filename, importance_maps=importance_maps, labels=labels)

def load_importance_data(filename=os.path.join(OUTPUT_DIR, 'importance_maps.npz')):
    data = np.load(filename)
    return data['importance_maps'], data['labels']

#-------------------------------------------------------------------------Save and Load Best Params------------------------------------------------------------

def save_best_params(params, filename=BEST_PARAMS_FILE):
    ensure_output_dir()
    with open(filename, 'w') as file:
        json.dump(params, file)

def load_best_params(filename=BEST_PARAMS_FILE):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return None
    
#--------------------------------------------------------------------------------Main--------------------------------------------------------------------------

def main():
    
    importance_map_file = os.path.join(OUTPUT_DIR, 'importance_maps.npz')
    
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

    best_params = load_best_params()
    
    if best_params:
        print(f'Using saved hyperparameters: {best_params}')
    else:
        # Hyperparameter search using HalvingGridSearchCVV
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'gamma': ['scale'],
            'kernel': ['linear', 'rbf'],
            'class_weight': ['balanced']
        }
        
        clf = svm.SVC()
        halving_search = HalvingGridSearchCV(clf, param_grid, cv=5, verbose=2, n_jobs=-1)
        halving_search.fit(importance_maps, labels)

        best_params = halving_search.best_params_
        print(f'Best Hyper-Parameters SVM: {best_params}')
        
        save_best_params(best_params)
    
    # Cross-validation
    label_mapping = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze', 3: 'Both'}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    
    for train_index, test_index in skf.split(importance_maps, labels):
        X_train, X_test = importance_maps[train_index], importance_maps[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        final_clf = svm.SVC(**best_params)
        final_clf.fit(X_train, y_train)
        y_pred = final_clf.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.values()))
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Importance Map (SVM) - Cross-Validation')
    ensure_output_dir()
    output_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_cv.png')
    plt.savefig(output_path)
    plt.close()      
        
    accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f'\nFinal SVM accuracy (across all folds): {accuracy:.4f}')    
        
if __name__ == '__main__':
    main()
