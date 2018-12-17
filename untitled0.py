#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset.
dataset=pd.read_csv('studentinfo.csv')
X=dataset.iloc[:,[0,1,2,3,4,5,6,7]].values
Y=dataset.iloc[:,8].values



#Checking if any column contains null values.
dataset.isnull().sum()


#Data Preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])
X[:,1]=le.fit_transform(X[:,1])
X[:,2]=le.fit_transform(X[:,2])
X[:,3]=le.fit_transform(X[:,3])
X[:,4]=le.fit_transform(X[:,4])
X[:,7]=le.fit_transform(X[:,7])
Y=le.fit_transform(Y)

#Encoding the data.
onehotencoder=OneHotEncoder(categorical_features=[0,2,3,4])
X=onehotencoder.fit_transform(X).toarray()


#Splitting the dataset into Train set and Test set.
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, Ytrain, Ytest=train_test_split(X, Y, test_size=0.30, random_state=0)

#Feature Scaling of dataset.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.fit_transform(Xtest)


#Fitting logistic regression to Training Data.
from sklearn.linear_model import LogisticRegression
classifier_lr=LogisticRegression(random_state=0)
classifier_lr.fit(Xtrain, Ytrain)

#Testing classifier on Test set.
Ypred_lr=classifier_lr.predict(Xtest)

#Confusion matrix for logistic regression.
from sklearn.metrics import confusion_matrix
cm_lr=confusion_matrix(Ytest,Ypred_lr)

#Applying k-fold cross validation.
from sklearn.model_selection import cross_val_score
accuracies_lr=cross_val_score(estimator = classifier_lr, X=Xtrain, y=Ytrain, cv= 10, n_jobs=-1) 
lr_accuracy=accuracies_lr.mean()*100
lr_var=accuracies_lr.std()*100

#Fitting SVM to Training Data.
from sklearn.svm import SVC
classifier_svm=SVC(kernel='rbf' , random_state=0)
classifier_svm.fit(Xtrain, Ytrain)

#Testing classifier on Test set.
Ypred_svm=classifier_svm.predict(Xtest)

#Confusion matrix for SVM.
from sklearn.metrics import confusion_matrix
cm_svm=confusion_matrix(Ytest,Ypred_svm)

#Applying k-fold cross validation.
from sklearn.model_selection import cross_val_score
accuracies_svm=cross_val_score(estimator = classifier_svm, X=Xtrain, y=Ytrain, cv= 10, n_jobs=-1) 
svm_accuracy=accuracies_svm.mean()*100
svm_var=accuracies_svm.std()*100


#fitting knn
from sklearn.neighbors import KNeighborsClassifier
classifier_knn=KNeighborsClassifier(n_neighbors=1000, metric ='minkowski' ,p = 2)
classifier_knn.fit(Xtrain, Ytrain)

#Testing classifier on Test set.
Ypred_knn=classifier_knn.predict(Xtest)

#Confusion matrix for knn.
from sklearn.metrics import confusion_matrix
cm_knn=confusion_matrix(Ytest,Ypred_knn)

#Applying k-fold cross validation.
from sklearn.model_selection import cross_val_score
accuracies_knn=cross_val_score(estimator = classifier_knn, X=Xtrain, y=Ytrain, cv= 10, n_jobs=-1) 
knn_accuracy=accuracies_knn.mean()*100
knn_var=accuracies_knn.std()*100

#fitting Decision Tree.
from sklearn.tree import DecisionTreeClassifier
classifier_dt=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_dt.fit(Xtrain, Ytrain)

#Testing classifier on Test set.
Ypred_dt=classifier_dt.predict(Xtest)

#Confusion matrix for Decision Tree.
from sklearn.metrics import confusion_matrix
cm_dt=confusion_matrix(Ytest,Ypred_dt)

#Applying k-fold cross validation.
from sklearn.model_selection import cross_val_score
accuracies_dt=cross_val_score(estimator = classifier_dt, X=Xtrain, y=Ytrain, cv= 10) 
dt_accuracy=accuracies_dt.mean()*100
dt_var=accuracies_dt.std()*100


#fitting randomforest.
from sklearn.ensemble import RandomForestClassifier
classifier_rf= RandomForestClassifier(n_estimators=500,  criterion='entropy', random_state=0, n_jobs=-1)
classifier_rf.fit(Xtrain, Ytrain)

#Testing classifier on Test set.
Ypred_rf=classifier_rf.predict(Xtest)

#Confusion matrix for Random Forest.
from sklearn.metrics import confusion_matrix
cm_rf=confusion_matrix(Ytest,Ypred_rf)

#Applying k-fold cross validation.
from sklearn.model_selection import cross_val_score
accuracies_rf=cross_val_score(estimator = classifier_rf, X=Xtrain, y=Ytrain, cv= 10, n_jobs=-1) 
rf_accuracy=accuracies_rf.mean()*100
rf_var=accuracies_rf.std()*100



#Visualising the Comparitive analysis of performance of different algorithms. 
objects = ('Logistic Regression','Support Vector Machine', 'K-Nearest Neighbours', 'Decision Tree', 'Random Forest')
x_pos = np.arange(len(objects))
performance = [lr_accuracy,svm_accuracy,knn_accuracy,dt_accuracy,rf_accuracy]
labels=[lr_accuracy,svm_accuracy,knn_accuracy,dt_accuracy,rf_accuracy]
plt.bar(x_pos, performance, align='center', alpha=0.8)
plt.xticks(x_pos, objects)
plt.ylabel('Accuracy in Percentage.')
plt.xlabel('Different Classification Models')
plt.title('Comparitive Analysis')
for i in range (len(x_pos)):
    plt.text(x= x_pos[i]-0.3, y= performance[i]+1, s=labels[i], size=10)

 
plt.show()

#Visualising the Comparitive analysis of Variance of different algorithms. 
objects2 = ('Logistic Regression','Support Vector Machine', 'K-Nearest Neighbours', 'Decision Tree', 'Random Forest')
x_pos2 = np.arange(len(objects2))
variance = [lr_var,svm_var,knn_var,dt_var,rf_var]
labels2=[lr_var,svm_var,knn_var,dt_var,rf_var]
plt.bar(x_pos2, variance, align='center', alpha=0.8)
plt.xticks(x_pos2, objects2)
plt.ylabel('Variance')
plt.xlabel('Different Classification Models')
plt.title('Comparitive Analysis')
for i in range (len(x_pos2)):
    plt.text(x= x_pos2[i]-0.3, y= variance[i]+0.0001, s=labels2[i], size=10)

 
plt.show()








