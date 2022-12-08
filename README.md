# TFusion: Transformer based N-to-One Fusion Block
  
Our implementation is on an NVIDIA RTX 3090 (24G) with PyTorch 1.8.1.

## Datasets
We use the BraTS2020 dataset, an open-source dataset.    
Please download and unzip the 'MICCAI_BraTS2020_TrainingData' into `./dataset`.  
Then, please `cd ./process` and run the following commands to prepare the data:
```
python split.py
```

## Training Examples
```
python train.py --phase train --model_name TF_RMBTS
```
Saved models can be found at `./checkpoint`. 
model_name includes :  'TF_U_Hemis3D', 'U_Hemis3D', 'RMBTS', 'TF_RMBTS', 'LMCR', 'TF_LMCR' .

Note that 'RMBTS' refers to 'FDGF'.

## Test Examples (Please train the model before test.)
```
python train.py --phase test --model_name TF_RMBTS
```
Brain tumor segmentation results for test data can be found at `./checkpoint`.  

## Evaluation
```
python evaluation.py --model_name TF_RMBTS

```



