#Step 1: Import Python Libraries
#Pandas and Numpy for numerical Processing

import pandas as pd
import numpy as np

# Schikit-Learn for machine Learning

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from sklearn.neighbors import KNeighborsClassifier
rom sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Matplotlib and searborn for graphs

import matplotlib.pyplot as plt
import seaborn as sns

# Joblib for saving our machine learning model

import joblib

Attacklabels = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'neptune', 'nmap', 'normal', 'phf', 'portsweep', 'satan', 'spy', 'warezclient']

# Step 2: Load dataset

pd_Ransomware = pd.read_csv('nsl_kdd_encoded.csv')
pd_Ransomware.dropna(axis= 0, how='any')

print(pd_Ransomware.head())

# Lets see how our dataset looks like

print(pd_Ransomware.describe())

# Crosstab visualization

table = pd.crosstab(pd_Ransomware.count, pd_Ransomware.classification_encoded).plot(kind='bar')

plt.title("Traffic count by Attack Type")
plt.ylabel("Traffic count")
plt.xlabel("Attack Type")

handles, labels = table.get_legend_handles_table()
print(handles, labels)
plt.legend(lables=Attacklabels)
plt.show()

pd_Ransomware.hist(column="classification_encoded")
plt.show()

#Step 3: Extract features and lables

np_RansomwareFeatures = pd_Ransomware.iloc[0:, :6].numpy()

# Extract the attack labels

malwareLables = pd_Ransomware.iloc[0:, -1].to_numpy()

# Step 4: Break the dataset into the train and test sets

X_train, X_test, y_train, y_test = train_test_split(np_RansomwareFeatures, malwareLables, test_size = 0.2, random_state = 42)

# Step5: Build a machine learning model

knn = KNeighborsClassifier(n_neighbors = 12).fit(X_train, y_train)

# Step 6: Test the model

accuracy = knn.score(X_test, y_test)
print(accuracy)

knn_predictions = knn.predict(X_test)
cr = classification_report(y_test, knn_predictions)
print(cr)

# Step 7: Save Your model

joblib.dump(knn, "KNN_model.pkl")

