#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import random


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean

import seaborn as sns

import itertools
from itertools import product

import csv

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

from sklearn.decomposition import KernelPCA

import timeit
# from fnvhash import fnv1a_32
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, matthews_corrcoef
from sklearn.metrics import precision_recall_curve

from sklearn.decomposition import PCA


print("done")



# # Loading Data

# In[2]:

data_name = "seq_data_7000"

seq_data_1 = np.load("/olga-data1/Sarwan/Spike7k/" + data_name + ".npy",allow_pickle=True)

seq_data = []
for i in range(len(seq_data_1)):
    seq_data.append(str(seq_data_1[i]).replace(", ","").replace("[","").replace("]","").replace("\'",""))

attribute_data = np.load("/olga-data1/Sarwan/Spike7k/seq_data_variant_names_7000.npy")


attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
print("Attribute data preprocessing Done")


# In[3]:


seq_data = np.array(seq_data)
seq_data.shape


# In[4]:


len(seq_data[0]),len(seq_data[1])


# # Bezier curve calculation

# In[ ]:


# Generate control points based on amirosetno acid mapping
def generate_control_points(sequence):
    control_points = {}
    for i, aa in enumerate(sequence):
        # Example: Use the index i as x-coordinate and ASCII value of aa as y-coordinate
        control_points[aa] = (i, ord(aa))
    return control_points

# Bezier curve calculation
def calculate_bezier_point(t, points):
    n = len(points) - 1
    result = np.zeros(2)
    for i, p in enumerate(points):
        coeff = np.math.comb(n, i) * (1 - t)**(n - i) * t**i
        result += coeff * np.array(p)
    return result


# protein_sequences = ["VVLHFVWPPQSFDNVCKWKSHIIDFSWFKIPHSLYMSQIPVIKYVLQCEMNHRAQGLAFLAKDNWNWNCDD",
#                    "VVLHFVWPPQSFDNVCKWKSHIIDFSWFKIPHSLYMSQIPVIKYVLQCEMNHRAQGLAFLAKDNW",
#                    "VVLHFVWPPQSFDNVCKWKSHIIDFSWFKIPHSLYMSQIPVIKYVLQCEMNHRAQGLAFLAK",
#                    "VVLHFVWPPQSFDNVCKWKSHIIDFSWFKIPHSLYMSQIPVIKYVLQCEMNHRAQGLAFLAKDNWNWNC",
#                    "VVLHFVWPPQSFDNVCKWKSHIIDFSWFKIPHSLYMSQIPVIKYVLQCEMNHRAQGLAFLAKDNWNWN"]

protein_sequences = seq_data[:]

embedding_final = []
for seqs in range(len(protein_sequences)):
    if seqs%10==0:
        print("Index: ",seqs,"/",len(protein_sequences))
    protein_sequence = protein_sequences[seqs]


    control_points = generate_control_points(protein_sequence)
    # Generate points along Bezier curves for each amino acid
    x_coords, y_coords = [], []
    num_points = 500  # Number of points along each curve
    t_values = np.linspace(0, 1, num_points)

    # Create figure and subplot



    #     y = np.cos(x) 
    # Vary control points to add randomness
    for aa in protein_sequence:
        if aa in control_points:
            original_point = control_points[aa]
            points = [original_point]
            for _ in range(3):
                deviation = np.random.uniform(-10, 10, size=2)  # Random deviation from original point
                modified_point = original_point + deviation
                points.append(modified_point)
            curve_points = np.array([calculate_bezier_point(t, points) for t in t_values])
            x_coords.extend(curve_points[:, 0])
            y_coords.extend(curve_points[:, 1])

    #########################
    # Create a matrix of curve points
    curve_points_matrix = np.array(list(zip(x_coords, y_coords)))
    # Flatten the matrix to a 1D array
    curve_points_1d = curve_points_matrix.flatten()
#         curve_points_matrix.shape
#         len(curve_points_1d)
    embedding_final.append(curve_points_1d)
    #########################


# # Data Padding

# In[ ]:


embedding_final = np.array(embedding_final)


max_len = 0
for i in range(len(embedding_final)):
    if len(embedding_final[i])>max_len:
        max_len = len(embedding_final[i])
        
embedding_final_padded = []

for i in range(len(embedding_final)):
    final_data_tmp = list(embedding_final[i])
    if len(final_data_tmp)<max_len:
        for j in range(len(final_data_tmp),max_len):
            final_data_tmp.append(0)
    embedding_final_padded.append(final_data_tmp)




# In[ ]:


embedding_final_padded = np.array(embedding_final_padded)
(embedding_final_padded.shape)


# In[ ]:


#########################################################
print("Now applying PCA")
from sklearn.decomposition import PCA
# initialize PCA with desired number of components
pca = PCA(n_components=500)
# apply PCA on 2D array
arr_2d_pca = np.array(pca.fit_transform(np.array(embedding_final_padded)))

# seq_data = arr_2d_pca[:]
##########################################################


# # Classification Functions

# In[ ]:


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)


    
# In[5]
##########################  SVM Classifier  ################################
def svm_fun(X_train,y_train,X_test,y_test):

    #scaler = RobustScaler()
    X_train = preprocessing.scale(X_train)  
    X_test = preprocessing.scale(X_test)  
    
    
    start = timeit.default_timer()
    
    
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_new = stop - start
#     print("NB Time : ", stop - start) 
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix SVM : \n", confuse)
    #print("SVM Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])


    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1], time_new]
#     check = [svm_acc,svm_prec,npv,sensitivity,specificity,mcc,svm_recall,svm_f1_weighted,svm_f1_macro,macro_roc_auc_ovo[1],roc_pr_auc, time_new]
    return(check)
    


# In[5]
##########################  NB Classifier  ################################
def gaus_nb_fun(X_train,y_train,X_test,y_test):
    start = timeit.default_timer()
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 
    
    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    stop = timeit.default_timer()
    time_new = stop - start


    NB_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Gaussian NB Accuracy:",NB_acc)

    NB_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Precision:",NB_prec)
    
    NB_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Recall:",NB_recall)
    
    NB_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB F1 weighted:",NB_f1_weighted)
    
    NB_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Gaussian NB F1 macro:",NB_f1_macro)
    
    NB_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Gaussian NB F1 micro:",NB_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix NB : \n", confuse)
    #print("NB Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')

    check = [NB_acc,NB_prec,NB_recall,NB_f1_weighted,NB_f1_macro,macro_roc_auc_ovo[1], time_new]
#     check = [NB_acc,NB_prec,npv,sensitivity,specificity,mcc,NB_recall,NB_f1_weighted,NB_f1_macro,macro_roc_auc_ovo[1],roc_pr_auc, time_new]

    return(check)

# In[5]
##########################  MLP Classifier  ################################
def mlp_fun(X_train,y_train,X_test,y_test):
    start = timeit.default_timer()
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 
    
    
    # Feature scaling
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test_2 = scaler.transform(X_test)


    # Finally for the MLP- Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
    mlp.fit(X_train, y_train)


    y_pred = mlp.predict(X_test_2)
    
    stop = timeit.default_timer()
    time_new = stop - start
    
    MLP_acc = metrics.accuracy_score(y_test, y_pred)
#     print("MLP Accuracy:",MLP_acc)
    
    MLP_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("MLP Precision:",MLP_prec)
    
    MLP_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("MLP Recall:",MLP_recall)
    
    MLP_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("MLP F1:",MLP_f1_weighted)
    
    MLP_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("MLP F1:",MLP_f1_macro)
    
    MLP_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("MLP F1:",MLP_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix MLP : \n", confuse)
    #print("MLP Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    

    check = [MLP_acc,MLP_prec,MLP_recall,MLP_f1_weighted,MLP_f1_macro,macro_roc_auc_ovo[1], time_new]
#     check = [MLP_acc,MLP_prec,npv,sensitivity,specificity,mcc,MLP_recall,MLP_f1_weighted,MLP_f1_macro,macro_roc_auc_ovo[1],roc_pr_auc, time_new]
    return(check)

# In[5]
##########################  knn Classifier  ################################
def knn_fun(X_train,y_train,X_test,y_test):
    start = timeit.default_timer()
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 
    
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    stop = timeit.default_timer()
    time_new = stop - start

    knn_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Knn Accuracy:",knn_acc)
    
    knn_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Knn Precision:",knn_prec)
    
    knn_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Knn Recall:",knn_recall)
    
    knn_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Knn F1 weighted:",knn_f1_weighted)
    
    knn_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Knn F1 macro:",knn_f1_macro)
    
    knn_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Knn F1 micro:",knn_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix KNN : \n", confuse)
    #print("KNN Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
  
    check = [knn_acc,knn_prec,knn_recall,knn_f1_weighted,knn_f1_macro,macro_roc_auc_ovo[1], time_new]
#     check = [knn_acc,knn_prec,npv,sensitivity,specificity,mcc,knn_recall,knn_f1_weighted,knn_f1_macro,macro_roc_auc_ovo[1],roc_pr_auc, time_new]
    return(check)

# In[5]
##########################  Random Forest Classifier  ################################
def rf_fun(X_train,y_train,X_test,y_test):
    start = timeit.default_timer()
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 
    
    
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 100)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    stop = timeit.default_timer()
    time_new = stop - start

    fr_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Random Forest Accuracy:",fr_acc)
    
    fr_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Random Forest Precision:",fr_prec)
    
    fr_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Random Forest Recall:",fr_recall)
    
    fr_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Random Forest F1 weighted:",fr_f1_weighted)
    
    fr_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Random Forest F1 macro:",fr_f1_macro)
    
    fr_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Random Forest F1 micro:",fr_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix RF : \n", confuse)
    #print("RF Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    

    check = [fr_acc,fr_prec,fr_recall,fr_f1_weighted,fr_f1_macro,macro_roc_auc_ovo[1], time_new]
#     check = [fr_acc,fr_prec,npv,sensitivity,specificity,mcc,fr_recall,fr_f1_weighted,fr_f1_macro,macro_roc_auc_ovo[1],roc_pr_auc, time_new]
    return(check)

# In[5]
    ##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train,y_train,X_test,y_test):
    start = timeit.default_timer()
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 
    

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    stop = timeit.default_timer()
    time_new = stop - start

    LR_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    LR_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    LR_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    LR_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    LR_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    LR_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix LR : \n", confuse)
    #print("LR Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
  

    check = [LR_acc,LR_prec,LR_recall,LR_f1_weighted,LR_f1_macro,macro_roc_auc_ovo[1], time_new]
#     check = [LR_acc,LR_prec,npv,sensitivity,specificity,mcc,LR_recall,LR_f1_weighted,LR_f1_macro,macro_roc_auc_ovo[1],roc_pr_auc, time_new]
    return(check)


def fun_decision_tree(X_train,y_train,X_test,y_test):
    from sklearn import tree
    
    start = timeit.default_timer()
#     stop = timeit.default_timer()
#     print("NB Time : ", stop - start) 
    
    clf = tree.DecisionTreeClassifier()    
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()
    time_new = stop - start
    
    dt_acc = metrics.accuracy_score(y_test, y_pred)    
    dt_prec = metrics.precision_score(y_test, y_pred,average='weighted')    
    dt_recall = metrics.recall_score(y_test, y_pred,average='weighted')    
    dt_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')    
    dt_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')    
    dt_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
    confuse = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix DT : \n", confuse)
    #print("DT Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    

    check = [dt_acc,dt_prec,dt_recall,dt_f1_weighted,dt_f1_macro,macro_roc_auc_ovo[1], time_new]
#     check = [dt_acc,dt_prec,npv,sensitivity,specificity,mcc,dt_recall,dt_f1_weighted,dt_f1_macro,macro_roc_auc_ovo[1],roc_pr_auc, time_new]
    return(check)




import timeit

np.save("/olga-data1/Sarwan/Bezier_Curve_Embedding/Data/Bezier_Embedding_" + data_name + "_After_PCA.npy")

X = np.array(arr_2d_pca)
y = np.array(int_hosts)

# print("Accuracy   Precision   Recall   F1 (weighted)   F1 (Macro)   F1 (Micro)   ROC AUC")
svm_table = []
gauu_nb_table = []
mlp_table = []
knn_table = []
rf_table = []
lr_table = []
dt_table = []

total_splits = 5

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=total_splits, test_size=0.3)
sss.get_n_splits(X, y)

for splits_ind in range(total_splits):
    train_index, test_index = next(sss.split(X, y)) 

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    start = timeit.default_timer()
    gauu_nb_return = gaus_nb_fun(X_train,y_train,X_test,y_test)
    stop = timeit.default_timer()
    print("NB Time : ", stop - start) 

    start = timeit.default_timer()
    mlp_return = mlp_fun(X_train,y_train,X_test,y_test)
    stop = timeit.default_timer()
    print("MLP Time : ", stop - start) 

    start = timeit.default_timer()
    knn_return = knn_fun(X_train,y_train,X_test,y_test)
    stop = timeit.default_timer()
    print("KNN Time : ", stop - start) 

    start = timeit.default_timer()
    rf_return = rf_fun(X_train,y_train,X_test,y_test)
    stop = timeit.default_timer()
    print("RF Time : ", stop - start) 

    start = timeit.default_timer()
    lr_return = lr_fun(X_train,y_train,X_test,y_test)
    stop = timeit.default_timer()
    print("LR Time : ", stop - start) 

    start = timeit.default_timer()
    dt_return = fun_decision_tree(X_train,y_train,X_test,y_test)
    stop = timeit.default_timer()
    print("DT Time : ", stop - start) 

    start = timeit.default_timer()
    svm_return = svm_fun(X_train,y_train,X_test,y_test)
    stop = timeit.default_timer()
    print("SVM Time : ", stop - start) 

    gauu_nb_table.append(gauu_nb_return)
    mlp_table.append(mlp_return)
    knn_table.append(knn_return)
    rf_table.append(rf_return)
    lr_table.append(lr_return)
    dt_table.append(dt_return)
    svm_table.append(svm_return)

    svm_table_final = DataFrame(svm_table, columns=["Accuracy","Precision", "Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime"])
    gauu_nb_table_final = DataFrame(gauu_nb_table, columns=["Accuracy","Precision", "Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime"])
    mlp_table_final = DataFrame(mlp_table, columns=["Accuracy","Precision", "Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime"])
    knn_table_final = DataFrame(knn_table, columns=["Accuracy","Precision", "Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime"])
    rf_table_final = DataFrame(rf_table, columns=["Accuracy","Precision", "Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime"])
    lr_table_final = DataFrame(lr_table, columns=["Accuracy","Precision", "Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime"])
    dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision", "Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC","Runtime"])
    


# In[ ]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.mean()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.mean()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.mean()))))
final_mean_mat.append(np.transpose((list(knn_table_final.mean()))))
final_mean_mat.append(np.transpose((list(rf_table_final.mean()))))
final_mean_mat.append(np.transpose((list(lr_table_final.mean()))))
final_mean_mat.append(np.transpose((list(dt_table_final.mean()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC", "Runtime"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

# final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision", "NPV","Sensitivity", "Specificity", 
#                                                     "MCC","Recall",
#                                                     "F1 (weighted)","F1 (Macro)","ROC AUC", "ROC-PR","Runtime"], 
#                           index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)
# final_avg_mat *= 100


# In[ ]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.std()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.std()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.std()))))
final_mean_mat.append(np.transpose((list(knn_table_final.std()))))
final_mean_mat.append(np.transpose((list(rf_table_final.std()))))
final_mean_mat.append(np.transpose((list(lr_table_final.std()))))
final_mean_mat.append(np.transpose((list(dt_table_final.std()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC", "Runtime"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

# final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision", "NPV","Sensitivity", "Specificity", 
#                                                     "MCC","Recall",
#                                                     "F1 (weighted)","F1 (Macro)","ROC AUC", "ROC-PR","Runtime"], 
#                           index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)


# In[ ]:


#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.max()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.max()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.max()))))
final_mean_mat.append(np.transpose((list(knn_table_final.max()))))
final_mean_mat.append(np.transpose((list(rf_table_final.max()))))
final_mean_mat.append(np.transpose((list(lr_table_final.max()))))
final_mean_mat.append(np.transpose((list(dt_table_final.max()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","ROC AUC", "Runtime"], 
                          index=["SVM","NB","MLP","KNN","RF","LR","DT"])

# final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision", "NPV","Sensitivity", "Specificity", 
#                                                     "MCC","Recall",
#                                                     "F1 (weighted)","F1 (Macro)","ROC AUC", "ROC-PR","Runtime"], 
#                           index=["SVM","NB","MLP","KNN","RF","LR","DT"])

print(final_avg_mat)
