# Reproducibility Project for CS598 DL4H in Spring 2022 
# Predicting Heart Failure Readmission from Clinical Notes Using Deep Learning
Xiong Liu, Yu Chen, Jay Bae, Hu Li, Joseph Johnston, and Todd Sanger. 2019. Predicting heart failure readmission from clinical notes using deep learning. In 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 2642–2648

This repository is the official implementation of this reproduction study (https://github.com/wecho3/team108). 
There is no publicly available code for the original paper. 

## Requirements

The code was developed on Python version 3.9.6, and it uses the following libraries.
Python native: os, time, datetime, string, pickle, gc
Non-native: pandas, numpy, keras, pytorch, nltk, gensim, sklearn

To install requirements:

```setup
pip install -upgrade pandas
pip install -upgrade numpy
pip install -upgrade keras
pip install -upgrade pytorch
pip install -upgrade nltk
pip install -upgrade gensim
pip install -upgrade sklearn
```

Pandas, keras, nltk, and gensim are used for data preprocessing of the MIMIC-III raw data. 
Numpy, pytorch, and sklearn are used in the implementation of the models.

>📋  Environment Setup and Database Downloads

The code is structured in a Jupyter notebook, so you will need access to open and run this file using a Jupyter web or standalone notebook or in an IDE such as Visual Studio Code with the Jupyter plugin. 

The following datasets are not available in the libraries and need to be obtained separately. 
MIMIC-III dataset for the healthcare data publicly available through PhysioNet (https://physionet.org/content/mimiciii/1.4/). This data needs to be downloaded in a folder named “mimic-iii-clinical-database-1.4” in the same directory as the code. 
Pre-Trained Word2Vec Model on PubMed abstracts and PubMed Central full-text documents publicly available through the University of Turku (https://bio.nlplab.org/). The model can be downloaded from clicking on the link in the “Word vectors” section and downloading “PubMed-and-PMC-w2v.bin”. This model needs to be downloaded into the same directory as the code. 

## Data Pre-Processing
The “MIMIC-III Data Processing” section of the Jupyter notebook needs to be run in order for data to be processed. At the end of the section, you will have the opportunity to save the dataframes to be used in the next section without going through the entire section again. Note, you will need to run the first cell to load all the libraries and need the downloads of the MIMC-III dataset and the pre-trained Word2Vec model in this section. 

After processing the data, run the “Dataset Creation” section of the notebook to create the datasets to be used in model training and evaluation. After section 2.1, you will be able to save the processed tensors for use in the dataset without running the previous sections again. 

Lastly for model creation, run the “CNN Model” section up to the end of section 3.2 to create the model definition and prepare the CNN model for training and evaluation. Similarly, run the “Random Forest” section up to the end of 4.3.2 to prepare the baseline random forest model for training and evaluation. 

## Training

To train the CNN model in the paper, run section 3.3 in the “CNN Model” section of the notebook. 
To train the random forest baseline model, run section 4.3.3 in the “Random Forest Model” section of the notebook. 

>📋  Hyperparameter Settings

For the CNN model, the hyperparameters are available to update at the top of the cell in section 3.3 Training the Model. The following are available as these variables: 
•	Epochs: (int) Training Epochs
•	N_features: (int) Features for the CNN convolution layers (all layers will have the same number of features for each of the 1, 2, and 3 kernel size one-dimensional convolutions)
•	Filter_layers: (1x3 array of int) Number of layers for each of the one-dimensional convolutions
•	Dropout_p: (float) Probability for the dropout layer
•	Learning_rate: (float) Learning rate set for the optimizer
In addition to these hyperparameters, some hyperparameters were fixed to match the settings mentioned in the original paper. These include the input channels (matches the embedding dimension of the pre-trained Word2Vec model), the kernel size (three different kernel sizes of 1, 2, and 3), and the stride (set to 1 to evaluate every word/word combination in the notes). 

For the random forest model, the hyperparameters are defined directly in the RandomForestClassifier class from the SKLearn library (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). The only hyperparameters defined in the original paper was to evaluate the maximum number of features from 10,000 to 25,000 with a step size of 5,000, which is set in this implementation. 
The following hyperparameters are available as parameters in the training/validation wrapper function:
•	Estimators: (int) defaults to 100 as in the SKLearn class. 
•	Depth: (int) defaults to None (as in the SKLearn class), which is unlimited depth
Other hyperparameters are available directly through the random forest classifier class. 

## Evaluation

The CNN model is evaluated in the same cell in section 3.3 Training the Model because of the 10-fold cross validation. Each of the folds has a new model instantiated, trained, and evaluated, and the printout will show the evaluation metrics for each training epoch, fold, and average across all folds. 

To evaluate the random forest baseline model, run section 4.3.3 Evaluate Random Forest. This cell trains the model and evaluates them as with the CNN model because of the 10-fold cross evaluation. 

## Pre-trained Models

There are no pre-trained models because the models need to be trained and evaluated separately in each cross-validation fold. When running the training and evaluation step in the sections above, the models will be trained and then evaluated in the course of running the code. All accurately predicted samples from the datasets are saved off locally for the chi-squared based interpretation step. 

## Results
The data preprocessing results in the following heart failure admission counts:
| | Number of Admissions  | Number of Admissions with Discharge Notes  |
| ------------- |:-------------:| :-----:|
| All heart failure admissions | 14,040 | 13,746 |
| Heart failure general readmissions | 3,604 | 3,543 |
| Heart failure 30 day readmissions | 969 | 962 |

Our CNN and random forest models achieve the following performance on the processed heart failure/heart failure readmission data from MIMIC-III:
| Task | Model | Precision | Recall | F1 | Accuracy
| ------------- |:-------------:| :-----:| :-----:| :-----:| :-----:|
| General Readmissions | CNN | 0.646 | 0.646 | 0.644 | 64.39%
| | Random Forest | 0.629 | 0.802 | 0.704 | 66.40%
| 30 Day Readmissions | CNN | 0.642 | 0.832 | 0.725 | 68.91%
| | Random Forest | 0.653 | 0.754 | 0.699 | 67.56%

The chi-squared test interpretation results in the following 20 highest scored features with their counts from the correctly predicted results of the CNN model:

General Readmission 
| Word | Chi-Score | Positive Sample Count (n=2289) | Negative Sample Count (n=2274) |
| ------------- |:-------------:| :-----:| :-----:| 
| tablet | 5602.05 | 44844 | 24855 |
| sig | 4701.55 | 28473 | 14173 |
| mg | 4300.27 | 42891 | 25515 |
| po | 3527.43 | 33243 | 19438 |
| one | 3238.99 | 25303 | 13905 |
| daily | 2553.22 | 29552 | 18337 |
| day | 2047.19 | 22255 | 13573 |
| expired | 1336.95 | 6 | 1346 |
| capsule | 1290.66 | 7766 | 3855 |
| times | 1195.63 | 9160 | 4999 |
| every | 1148.57 | 7564 | 3898 |
| hd | 1030.33 | 2651 | 764 |
| dialysis | 884.47 | 1920 | 461 |
| date/time | 864.15 | 2199 | 627 |
| insulin | 855.23 | 3799 | 1627 |
| chewable | 839.38 | 2068 | 571 |
| chronic | 817.53 | 5852 | 3115 |
| needed | 808.1 | 7134 | 4086 |
| provider | 752.39 | 2258 | 745 |
| inhalation | 685.86 | 2383 | 877 |

30 Day Readmission 
| Word | Chi-Score | Positive Sample Count (n=2289) | Negative Sample Count (n=2274) |
| ------------- |:-------------:| :-----:| :-----:| 
| tablet | 2776.03 | 12860 | 5672 |
| sig | 2177.17 | 8320 | 3284 |
| po | 1862.83 | 9893 | 4672 |
| daily | 1692.19 | 9093 | 4318 |
| mg | 1527.41 | 11608 | 6355 |
| one | 1407.81 | 7285 | 3398 |
| needed | 626.09 | 2306 | 889 |
| every | 615.03 | 2343 | 923 |
| capsule | 608.58 | 2224 | 853 |
| day | 605.19 | 6012 | 3593 |
| blood | 495.38 | 7660 | 5132 |
| name | 458.1 | 6547 | 4308 |
| hd | 455.06 | 769 | 129 |
| times | 382.15 | 2496 | 1290 |
| p.o | 370.15 | 191 | 794 |
| expired | 362.65 | 2 | 368 |
| ml | 357.83 | 1097 | 371 |
| inhalation | 355.57 | 781 | 192 |
| tid | 339.58 | 1105 | 391 |
| last | 321.09 | 4732 | 3136 |


