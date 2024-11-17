# Interpreting RespireNet

Implementation of the _‘Explaining Audio Classification models for Respiratory Sounds’_ project based on _RespireNet_. 

## Dependencies:
```
Python3.12
tensorflow==2.17.0
torch==2.4.0+cu124
torchvision==0.19.0+cu124
torchaudio==2.4.0+cu124
numpy==1.26.3
pandas==2.2.2
scikit-learn==1.5.1
scipy==1.14.1
matplotlib==3.9.2
seaborn==0.12.0
captum==0.7.0
shap==0.45.1
tensorboard==2.17.1
nltk==3.9.1
spacy==3.6.0
transformers==4.33.0
librosa==0.10.2.post1
pytorch-lightning==2.0.0
umap-learn==0.5.6
```

## Dataset
The dataset used is [ICBHI 2017 Challenge Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge). 


Here are the steps to follow to use the dataset. 
* Go to the official page of [ICBHI 2017 Challenge Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
* Download it and paste it into the folder: ```data```

## Post-hoc explainability command
The following command allows _Importance-Maps, Aggregated-GradCAM++, SmoothGradCam++_ to be generated on the entire _dataset_.  
```
python posthoc.py --data_dir ./data/icbhi_dataset/audio_text_data/ --checkpoint models/ckpt_best.pkl --folds_file ./data/patient_list_foldwise.txt --output_dir ./xai_results/ --sample_index $(seq 0 1443) | tee experiment.log
```
The following command allows _Importance-Maps, Aggregated-GradCAM++, SmoothGradCam++_ to be generated on the individual sample or a list of samples for possible future development. 
```
python posthoc.py --data_dir ./data/icbhi_dataset/audio_text_data/ --checkpoint models/ckpt_best.pkl --folds_file ./data/patient_list_foldwise.txt --output_dir ./xai_results/ --sample_index 0 | tee experiment.log
```
