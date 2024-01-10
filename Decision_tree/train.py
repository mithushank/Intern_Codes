from sklearn import datasets
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
#from DT import DecisionTree
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

group = clf.group(predictions,X)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print('Accuracy is ',acc)   
#confusion matrix
target_names = np.unique(y)
cm = confusion_matrix(y_test, predictions, labels=target_names,normalize='all')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot()
plt.show()
