import os
import math
os.chdir('C:\\Users\\LEGION\\OneDrive\\IT Onedrive\\PHD\\Paper\\Ali Paper\\PONE\\Revise\\Code\\GA-XGBoost')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from GA import genetic_algorithm

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_excel("C:\\Users\\LEGION\\OneDrive\\IT Onedrive\\PHD\\Paper\\Ali Paper\\PONE\\Revise\\Data\\data_balanced_SMOTE.xlsx")

# Preprocess the data
target_column = 'on_pump'
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# Running the Genetic Algorithm
best_solution, best_scores, best_scores_over_time = genetic_algorithm(X_train, y_train, X_test, y_test, 250, 20, 10, 0.1)
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
