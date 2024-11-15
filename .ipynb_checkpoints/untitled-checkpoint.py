# STEPS FOR MACHINE LEARNING
## Step 0: Import the necessary libraries
import numpy as np
from sklearn.model_selection import GridSearchCV
import scipy.optimize as opt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, classification_report, confusion_matrix
import itertools




### Function for plotting the Data
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')






## Step 1: Import Data
try:
    file = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("File Not Found")

print(file.columns)
file.head(5)




## Step 2: Clean the Data
print(file.shape)
print(file.dtypes, '\n')
print(file.isnull().sum())






## Step 3: Split the Data into Training/testing
X = np.asanyarray(file[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
y = np.asanyarray(file[['Class']])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)




## Step 4: Create a Model and  Step 5: Train the Model
# **: model = SVC(kernel='rbf', class_weight='balanced')
model = SVC(kernel='rbf') # I get bad resualt and because of My bad CPU I cannot run ** code
model.fit(X_train, y_train)




## Step 6: Make Predictions
predicted_y = model.predict(X_test)

## Check if predictions are made correctly
if len(predicted_y) != len(y_test):
    print("Mismatch in length between predictions and actual labels.")
    exit()


##Compute Confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_y, labels=[0, 1])
np.set_printoptions(precision=2)

print(classification_report(y_test, predicted_y))

##plot Confusion matrix(non-normalized)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["fail", "succeeded"], normalize=False, title="Confusion matrix")


print(f"f1_score: {f1_score(y_test, predicted_y, average='weighted')}")
print(f"jaccard_score: {jaccard_score(y_test, predicted_y, pos_label=1)}")


## Step 7: Evaluation and Improve
print(f"f1_score: {f1_score(y_test, predicted_y, average='weighted')}")
print(f"jaccard_score: {jaccard_score(y_test, predicted_y, pos_label=1)}")