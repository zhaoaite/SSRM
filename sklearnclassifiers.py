#!/usr/bin/python
# -*- coding: utf-8 -*- 
import time  
import numpy as np
from sklearn import metrics  
import pickle as pickle  
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
#from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
  
  
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier(n_neighbors=4)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=10)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=10)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in list(best_parameters.items()):  
        print(para, val)  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  

# SVM Classifier  
def MLP_classifier(train_x, train_y):      
    from sklearn.neural_network import MLPClassifier
    model =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30,20), random_state=1)
    model.fit(train_x,train_y)
    return model

  
def read_data():  
#    n_input = 16
#    n_steps = 20
#    m,n=a.shape
    a = np.loadtxt('result/ma_mlp.csv') 
    print(a.shape)
    #a = a.reshape(n_input*n_steps,-1)
#    a1 = np.loadtxt('./data/sleep/Bioradiolocation/Insomniac_data.csv') 
#    a=np.c_[a,a1]
#    a = a.reshape(-1,n_input*n_steps)
    
#    np.savetxt("data/sleep/Bioradiolocation/Insomniac_data1.csv",a)
#    newlist=np.ones((m,n))
#    for index1,i in enumerate(a):
#        for index2,j in enumerate(i):
#            newlist[index1,index2]=float(j)
#            print(newlist[index1,index2])
#    print(newlist.shape)
#    np.savetxt('tr.csv',newlist)
#    a=preprocessing.normalize(a, norm='l2') 
#    a1 = np.load('/home/zat/zresearch/SpyderNet/data/casia/casia.npy') 
    b=np.loadtxt("data/sleep/pressure/matr_label.csv")
#    #wake-sleep
#    wake_sleep_label=[]
#    for i in b:
#        if i==0:
#            wake_sleep_label.append(i)
#        else:
#            wake_sleep_label.append(1)
#    
#    b=wake_sleep_label
    
    #wake-REM-NREM
#    wake_REM_label=[]
#    for i in b:
#        if i==0:
#            wake_REM_label.append(i)
#        else:
#            if i==4:
#                wake_REM_label.append(2)
#            else:
#                wake_REM_label.append(1)
#    b=wake_REM_label 
    
#    #wake-REM-NREM 4class
#    fourclass_label=[]
#    for i in b:
#        if i==0:
#            fourclass_label.append(i)
#        else:
#            if i==4:
#                fourclass_label.append(3)
#            else: 
#                if i==3:
#                    fourclass_label.append(2)
#                else:
#                    fourclass_label.append(1)
#                
#    b=fourclass_label   
             
    print(len(b))
#    a=np.c_[a,a1]
    train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=0.3)
    print(train_x.shape)
    return train_x, train_y, test_x, test_y
      
if __name__ == '__main__':  
    thresh = 0.5  
    model_save_file = None  
    model_save = {}  
   
    test_classifiers = ['KNN','LR','DT','RF','DT','GBDT']  
    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier,  
                  'MLP':MLP_classifier
    }  
      
    print('reading training and testing data...')  
    train_x, train_y, test_x, test_y = read_data()  
      
    for classifier in test_classifiers:  
        print('******************* %s ********************' % classifier)  
        start_time = time.time()  
        model = classifiers[classifier](train_x, train_y)  
        print('training took %f s!' % (time.time() - start_time))  
        predict = model.predict(test_x)  
        score = model.predict_proba(test_x)  
#        np.savetxt('result/motion/classifier/PSG_wake_REM_score_'+classifier+'.csv',score)
#        np.savetxt('result/motion/classifier/PSG_wake_REM_score_test_y_'+classifier+'.csv',test_y)
        if model_save_file != None:  
            model_save[classifier] = model	
        precision = metrics.precision_score(test_y, predict,average='micro')  
        recall = metrics.recall_score(test_y, predict,average='micro')  
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))  
        accuracy = metrics.accuracy_score(test_y, predict)  
        print('accuracy: %.2f%%' % (100 * accuracy)) 
        # Compute confusion matrix
#        cnf_matrix = confusion_matrix(test_y, predict)
#        np.set_printoptions(precision=2)
#        # Plot non-normalized confusion matrix
##        plt.figure()
##        plot_confusion_matrix(cnf_matrix, classes=[],
##                              title='Confusion matrix, without normalization')
#        
#        # Plot normalized confusion matrix
#        
#        plt.figure()
##        cnf_matrix=np.array([[1,0.,0.,0.],[0.,0.9655,0.0345,0.],[0.,0.0166,0.9834,0.],[0.,0.0042,0.0112,0.9846]])
#        plot_confusion_matrix(cnf_matrix, classes=['CO','ALS','HD','PD'], normalize=True,
#                              title='Normalized confusion matrix')
#        
#        plt.show()
  
    if model_save_file != None:  
        pickle.dump(model_save, open(model_save_file, 'wb')) 
