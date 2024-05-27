# 8.)Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions. Java/Python ML library classes can be used for this problem.

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1- load data
iris = datasets.load_iris()

# 2 - split train and test data
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)

# 3 - train & predict data in KNN
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# 4 - Print Data Classification
print("Results of classification usin knn with k=1")
for r in range(0, len(x_test)):
    print("Sample :", str(x_test[r]), "Actual - Label", str(y_test[r]), "Predicted - Label", str(y_pred[r]))

# 5 - print accuracy, classificationReport, confusionMatrix
print("Classifictaion Accuracy :", classifier.score(x_test, y_test))
print("Accuracy Metrices :")
print(classification_report(y_test, y_pred))
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_pred))

# OUTPUT :
# Iris dataset is loaded...
# Dataset is split into training and testing..
# Size of training data and it's label (135, 4) (135,)
# Size of test data and it's label (15, 4) (15,)
# Label 0 - setosa
# Label 1 - versicolor
# Label 2 - virginica
# Results of classification usin knn with k=1
# Sample : [5.4 3.4 1.7 0.2] Actual - Label 0 Predicted - Label 0
# Sample : [6.9 3.1 4.9 1.5] Actual - Label 1 Predicted - Label 1
# Sample : [5.  3.5 1.3 0.3] Actual - Label 0 Predicted - Label 0
# Sample : [6.1 2.6 5.6 1.4] Actual - Label 2 Predicted - Label 2
# Sample : [6.7 2.5 5.8 1.8] Actual - Label 2 Predicted - Label 2
# Sample : [6.3 2.3 4.4 1.3] Actual - Label 1 Predicted - Label 1
# Sample : [4.9 3.1 1.5 0.1] Actual - Label 0 Predicted - Label 0
# Sample : [5.2 2.7 3.9 1.4] Actual - Label 1 Predicted - Label 1
# Sample : [5.1 2.5 3.  1.1] Actual - Label 1 Predicted - Label 1
# Sample : [5.6 2.8 4.9 2. ] Actual - Label 2 Predicted - Label 2
# Sample : [5.8 2.7 4.1 1. ] Actual - Label 1 Predicted - Label 1
# Sample : [4.8 3.  1.4 0.1] Actual - Label 0 Predicted - Label 0
# Sample : [5.4 3.7 1.5 0.2] Actual - Label 0 Predicted - Label 0
# Sample : [5.4 3.9 1.7 0.4] Actual - Label 0 Predicted - Label 0
# Sample : [6.  2.2 4.  1. ] Actual - Label 1 Predicted - Label 1
# Classifictaion Accuracy : 1.0
# Accuracy Metrices :
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         6
#            1       1.00      1.00      1.00         6
#            2       1.00      1.00      1.00         3

#     accuracy                           1.00        15
#    macro avg       1.00      1.00      1.00        15
# weighted avg       1.00      1.00      1.00        15

# Confusion Matrix :
# [[6 0 0]
#  [0 6 0]
#  [0 0 3]]
