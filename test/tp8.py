from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)

classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print("results of classifier with neighbos = 1")
for r in range(0, len(x_test)):
    print("sample", str(x_test[r]) , "actual label", str(y_test[r]),"predicted label", str(y_pred[r]))

print("classification report", classifier.score(x_test, y_test))
print("accuracy metrices \n", classification_report(y_test, y_pred))
print("confusion matrix \n", confusion_matrix(y_test, y_pred))