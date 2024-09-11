import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from mpl_toolkits.mplot3d import Axes3D

def loading_preprocessing():
    df1 = pd.read_csv('data/Gas1.csv')
    df2 = pd.read_csv('data/Gas2.csv')
    df3 = pd.read_csv('data/Gas3.csv')
    df4 = pd.read_csv('data/Gas4.csv')
    df5 = pd.read_csv('data/Gas5.csv')
    df = pd.concat([df1, df2, df3, df4, df5], axis=0)
    df = df.drop('Concentration', axis=1)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffling
    y = df['Type']
    X = df.drop('Type', axis=1)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

def scaling_PCA(X_train, X_test, y_train):
    # Standardizing the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Applying PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Creating a DataFrame from PCA components
    pca_df_train = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df_train['label'] = y_train.values
    return pca_df_train, X_train_pca, X_test_pca, pca

def plot_train_dataset(pca_df_train):
    # Plotting scatter plot of PCA applied dataset
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for label in np.unique(pca_df_train['label']):
        ax.scatter(pca_df_train.loc[pca_df_train['label'] == label, 'PC1'],
                   pca_df_train.loc[pca_df_train['label'] == label, 'PC2'],
                   pca_df_train.loc[pca_df_train['label'] == label, 'PC3'],
                   label=f'Class {label}', edgecolor='k')

    ax.set_title('PCA Applied Dataset Scatter Plot (Train Set)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()
    
    # Return the figure for Streamlit
    return fig

def show_variance_ratio(pca):
    explained_variance_ratio = pca.explained_variance_ratio_

    # Plotting explained variance ratio
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='b', alpha=0.7)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Explained Variance Ratio of Principal Components')
    ax.set_xticks(range(1, len(explained_variance_ratio) + 1))
    
    # Return the figure for Streamlit
    return fig

def add_noise(mu, sigma, X_train_pca):
    noise = np.random.normal(mu, sigma, X_train_pca.shape)
    X_train_pca = X_train_pca + noise
    return X_train_pca

# Other model functions (LR_accuracy, KNN_accuracy, etc.) remain the same...

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# returning training time , cross-validation time , testing time ,Cross-validation scores and average Cross-validation scores for each of the ML model applied. 
def LR_accuracy(X_train_pca, y_train, X_test_pca) :
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    # Define the number of folds for cross-validation
    k_folds = 5
    
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    log_reg = LogisticRegression(max_iter=1000)  # Increase max_iter
    
    # Measure training time
    start_train = time.time()
    log_reg.fit(X_train_scaled, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Perform K-fold cross-validation
    start_cv = time.time()
    cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=kf)
    end_cv = time.time()
    cv_time = end_cv - start_cv
    
    # Predict testing data and measure testing time
    start_test = time.time()
    predictions = log_reg.predict(X_test_scaled)
    end_test = time.time()
    test_time = end_test - start_test
    return train_time , cv_time , test_time , cv_scores , cv_scores.mean()
from sklearn.neighbors import KNeighborsClassifier

def KNN_accuracy(X_train_pca, y_train, X_test_pca):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    # Define the number of neighbors for KNN
    k_neighbors = 5
    
    # Create a KNN classifier with k=5
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    
    # Define the number of folds for cross-validation
    k_folds = 5
    
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Measure training time
    start_train = time.time()
    knn.fit(X_train_scaled, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Perform K-fold cross-validation
    start_cv = time.time()
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=kf)
    end_cv = time.time()
    cv_time = end_cv - start_cv
    
    # Predict testing data and measure testing time
    start_test = time.time()
    predictions = knn.predict(X_test_scaled)
    end_test = time.time()
    test_time = end_test - start_test
    
    return train_time, cv_time, test_time, cv_scores, cv_scores.mean()


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

def NB_accuracy(X_train_pca, y_train, X_test_pca):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    # Create a Gaussian Naive Bayes classifier
    nb_classifier = GaussianNB()
    
    # Define the number of folds for cross-validation
    k_folds = 5
    
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Measure training time
    start_train = time.time()
    nb_classifier.fit(X_train_scaled, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Perform K-fold cross-validation
    start_cv = time.time()
    cv_scores = cross_val_score(nb_classifier, X_train_scaled, y_train, cv=kf)
    end_cv = time.time()
    cv_time = end_cv - start_cv
    
    # Predict testing data and measure testing time
    start_test = time.time()
    predictions = nb_classifier.predict(X_test_scaled)
    end_test = time.time()
    test_time = end_test - start_test
    
    return train_time, cv_time, test_time, cv_scores, cv_scores.mean()

# Example of how to use this function:
# train_time, cv_time, test_time, cv_scores, avg_cv_score = NB_accuracy(X_train_pca, y_train, X_test_pca)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import time

def RF_accuracy(X_train_pca, y_train, X_test_pca):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    # Create a Random Forest classifier with 100 trees
    rf_classifier = RandomForestClassifier(n_estimators=100)
    
    # Define the number of folds for cross-validation
    k_folds = 5
    
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Measure training time
    start_train = time.time()
    rf_classifier.fit(X_train_scaled, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Perform K-fold cross-validation
    start_cv = time.time()
    cv_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=kf)
    end_cv = time.time()
    cv_time = end_cv - start_cv
    
    # Predict testing data and measure testing time
    start_test = time.time()
    predictions = rf_classifier.predict(X_test_scaled)
    end_test = time.time()
    test_time = end_test - start_test
    
    return train_time, cv_time, test_time, cv_scores, cv_scores.mean()

# Example of how to use this function:
# train_time, cv_time, test_time, cv_scores, avg_cv_score = RF_accuracy(X_train_pca, y_train, X_test_pca)

from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import time

def XGB_accuracy(X_train_pca, y_train, X_test_pca):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    # Adjust the target labels if necessary (e.g., if starting from 1)
    y_train_adjusted = y_train - 1
    
    # Create an XGBoost classifier
    xgb_classifier = XGBClassifier()
    
    # Define the number of folds for cross-validation
    k_folds = 5
    
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Measure training time
    start_train = time.time()
    xgb_classifier.fit(X_train_scaled, y_train_adjusted)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Perform K-fold cross-validation
    start_cv = time.time()
    cv_scores = cross_val_score(xgb_classifier, X_train_scaled, y_train_adjusted, cv=kf)
    end_cv = time.time()
    cv_time = end_cv - start_cv
    
    # Predict testing data and measure testing time
    start_test = time.time()
    predictions = xgb_classifier.predict(X_test_scaled)
    end_test = time.time()
    test_time = end_test - start_test
    
    return train_time, cv_time, test_time, cv_scores, cv_scores.mean()

# Example of how to use this function:
# train_time, cv_time, test_time, cv_scores, avg_cv_score = XGB_accuracy(X_train_pca, y_train, X_test_pca)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import time

def LDA_accuracy(X_train_pca, y_train, X_test_pca):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    # Create an LDA classifier
    lda_classifier = LinearDiscriminantAnalysis()
    
    # Define the number of folds for cross-validation
    k_folds = 5
    
    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Measure training time
    start_train = time.time()
    lda_classifier.fit(X_train_scaled, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Perform K-fold cross-validation
    start_cv = time.time()
    cv_scores = cross_val_score(lda_classifier, X_train_scaled, y_train, cv=kf)
    end_cv = time.time()
    cv_time = end_cv - start_cv
    
    # Predict testing data and measure testing time
    start_test = time.time()
    predictions = lda_classifier.predict(X_test_scaled)
    end_test = time.time()
    test_time = end_test - start_test
    
    return train_time, cv_time, test_time, cv_scores, cv_scores.mean()

# Example of how to use this function:
# train_time, cv_time, test_time, cv_scores, avg_cv_score = LDA_accuracy(X_train_pca, y_train, X_test_pca)
