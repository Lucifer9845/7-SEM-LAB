from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as split

dataset = load_iris()


xtrain, xtest, ytrain, ytest = split(dataset.data, dataset.target, random_state=1, test_size=0.2)

nb = GaussianNB()
classifier = nb.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)

print("classification report: ", metrics.classification_report(ytest, ypred))
print("accuracy score: ", metrics.accuracy_score(ytest, ypred))
print("confusion matrix: ", metrics.confusion_matrix(ytest, ypred))