from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from RandomForest import RandomForest
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


n_tree_list = [3]
e_time = []
acc_list = []
#timer Start
for n in n_tree_list:
    start_time = time.time()

    data = datasets.load_digits()
    X = data.data
    y = data.target
    df = pd.DataFrame(X,y)
    print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    clf = RandomForest(n_trees=n)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    #timer end
    end_time = time.time()
    elapsed_time = end_time - start_time
    e_time.append(elapsed_time)

    acc =  accuracy(y_test, predictions)
    acc_list.append(acc)
    print(acc)
    print(f"Time taken: {elapsed_time} seconds")

print()
#confusion matrix
# target_names = np.unique(y)
# cm = confusion_matrix(y_test, predictions, labels=target_names,normalize='all')
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
# disp.plot()
# plt.show()

plt.style.use('_mpl-gallery')

# make data
x1 = np.arange(0,len(e_time))
x2 = np.arange(0,len(acc_list))
y1 = e_time
y2 = acc_list

# plot
fig, ax1 = plt.subplots()  # Use a single set of axes (ax1)

# Plot the first data on the left y-axis
ax1.plot(x1, y1, color='blue', label='Elapsed Time', linewidth=2.0)
ax1.set_xlabel('number of Trees')
ax1.set_ylabel('Elapsed Time', color='blue')
ax1.tick_params('y', colors='blue')

# Create a second y-axis on the right side
ax2 = ax1.twinx()
ax2.plot(x2, y2, color='red', label='Accuracy Score', linewidth=2.0)
ax2.set_ylabel('Accuracy Score', color='red')
ax2.tick_params('y', colors='red')

# Title for the entire plot
plt.title('Elapsed Time and Accuracy Score')

# Show the plot
plt.show()


