import os
import math
os.chdir('C:\\Users\\LEGION\\OneDrive\\IT Onedrive\\PHD\\Paper\\Ali Paper\\PONE\\Revise\\Code\\SA-CB')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from SA import simulated_annealing

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_excel("C:\\Users\\LEGION\\OneDrive\\IT Onedrive\\PHD\\Paper\\Ali Paper\\PONE\\Revise\\Data\\data_balanced_SMOTE.xlsx")

# Preprocess the data
target_column = 'On_Pump'
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# pca = PCA(0.95)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# X_train = pd.DataFrame(X_train_pca, index=X_train.index, columns=[f'component_{i+1}' for i in range(X_train_pca.shape[1])])
# X_test = pd.DataFrame(X_test_pca, index=X_test.index, columns=[f'component_{i+1}' for i in range(X_test_pca.shape[1])])

# Running the Simulated Annealing algorithm
best_solution, best_scores, best_scores_over_time = simulated_annealing(X_train, y_train, X_test, y_test, max_iterations=250, temp=100)
print("Best Solution:", best_solution)
print(f"Final Best Scores: Accuracy = {best_scores[0]}, Precision = {best_scores[1]}, Recall = {best_scores[2]}, F1 Score = {best_scores[3]}, Sensitivity = {best_scores[4]}, Specificity = {best_scores[5]}, pr_auc = {best_scores[6]}, roc_auc = {best_scores[7]}")

# Plotting
plt.plot(best_scores_over_time, linewidth=2)
plt.title('Optimization Plot')
plt.xlabel('Iteration')
plt.ylabel('Best Score')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 600})
plt.savefig('output.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig('output.svg', format='svg', bbox_inches='tight')
plt.show()