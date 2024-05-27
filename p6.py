# 6.)Write a program to implement the naive Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split as split

# 1 - load data
dataset = load_iris()
x = dataset.data
y = dataset.target

# 2- split train test
x_train, x_test, y_train, y_test = split(x, y, test_size=0.2, random_state=1)

# 3 - train & predict
gnb = GaussianNB()
classifier = gnb.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print("Accuracy Metrices :", metrics.classification_report(y_test, y_pred))
print("The Acccuracy of Metrices is :", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, y_pred))

# OUTPUT :
# Accuracy Metrices :  precision    recall  f1-score   support

#            0       1.00      1.00      1.00        11
#            1       1.00      0.92      0.96        13
#            2       0.86      1.00      0.92         6

#     accuracy           0.97        30
#    macro avg       0.95      0.97      0.96        30
# weighted avg       0.97      0.97      0.97        30
# The Acccuracy of Metrices is : 0.9666666666666667
# Confusion Matrix
# [[11  0  0]
#  [ 0 12  1]
#  [ 0  0  6]]
