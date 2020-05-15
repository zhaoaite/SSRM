# SSRM
Self-Supervised Sleep Recognition Model

This is the code for the paper in title "Self-Supervised Learning from Multi-Sensor Data for Sleep Recognition", IEEE ACCESS  
Authors: Aite Zhao, Junyu Dong and Huiyu Zhou

# Abstract
Sleep recognition refers to detection or identification of sleep posture, sleep state or sleep stage, which can provide critical information for the diagnosis of sleep diseases. Most of sleep recognition methods are limited to single-task recognition, which only involves recognition tasks without pre-training based on single-modal sleep data, and there is no generalized model for multi-task recognition on multi-sensor sleep data. Moreover, the shortage and imbalance of sleep samples also limits the expansion of the existing machine learning methods like support vector machine, decision tree and convolutional neural network, which lead to the decline of the learning ability and over-fitting. It is also difficult to extract the temporal features of the continuous sleep data by only using spacial feature extraction approaches and nonlinear classifiers. Self-supervised learning technologies have shown their capabilities to learn significant feature representations. In this paper, a novel self-supervised learning model is proposed for sleep recognition, which is composed of an upstream self-supervised pre-training task and downstream recognition task. The upstream task is conducted to increase the data capacity, and the information of frequency domain and the rotation view are used to learn the multi-dimensional sleep feature representations. The downstream task is undertaken to fuse bidirectional long-short term memory and conditional random field as the sequential data recognizer to produce the sleep labels. Our experiments shows that our proposed algorithm provide promising results in sleep identification and can further be applied in clinical and smart home environments as a diagnostic tool with other available automated sleep monitoring systems.



# Datesets
The sleep dataset can be downloaded in 
 
Sleep Bioradiolocation Database: https://www.physionet.org/content/sleepbrl/1.0.0/  
PSG: https://www.physionet.org/content/sleep-accel/1.0.0/   
Pressure Map Dataset: https://www.physionet.org/content/pmd/1.0.0/  
Please refer to the preprocessing and other details on these three datasets.  

# Requirements
+ python >= 3.5
+ numpy >= 1.18.0
+ scipy
+ tensorflow


Other dependencies can be installed using the following command:
```
pip install -r requirements.txt
```

# Usage
Pretraining process:
```
python MLP_Workers.py
```

Recognition process:
```
python downstream_with_crf.py
```

# Citation
If you use these models in your research, please cite:

@article{Zhao2020,  
	author = {Aite Zhao and Junyu Dong and Huiyu Zhou},  
	title = {Self-Supervised Learning from Multi-Sensor Data for Sleep Recognition},  
	journal = {IEEE ACCESS},  
	year = {2020}  
}  
