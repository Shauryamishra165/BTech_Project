import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from main import (
    loading_preprocessing,
    scaling_PCA,
    plot_train_dataset,
    show_variance_ratio,
    LR_accuracy,
    KNN_accuracy,
    NB_accuracy,
    RF_accuracy,
    XGB_accuracy,
    LDA_accuracy,
    add_noise
)

# Streamlit app layout
st.title("BTP Machine Learning")

# Default data load and preprocessing
X_train_default, X_test_default, y_train_default, y_test_default = loading_preprocessing()
pca_df_train_default, X_train_pca_default, X_test_pca_default, pca_default = scaling_PCA(X_train_default, X_test_default, y_train_default)

# Tabs for different sections
tabs = st.tabs(["Upload Data", "PCA Plot", "Explained Variance Ratio", "Model Evaluation"])

# Initialize variables to hold data across tabs
uploaded_data = None
X_train = X_train_default
X_test = X_test_default
y_train = y_train_default
y_test = y_test_default
pca_df_train = pca_df_train_default
X_train_pca = X_train_pca_default
X_test_pca = X_test_pca_default
pca = pca_default

# Tab for uploading and viewing data
with tabs[0]:
    st.header("Upload and View Data")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(uploaded_data.head())
        
        if 'Type' in uploaded_data.columns:
            # Separate features and labels
            X_train = uploaded_data.drop('Type', axis=1)
            y_train = uploaded_data['Type']
            
            # Standardize and apply PCA to the uploaded data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            pca = PCA(n_components=3)
            X_train_pca = pca.fit_transform(X_train_scaled)

            # Create a DataFrame for PCA results
            pca_df_train = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2', 'PC3'])
            pca_df_train['label'] = y_train.values
            
        else:
            st.error("The uploaded data must contain a 'Type' column.")
    else:
        st.write("Using default dataset.")

# Tab for PCA plot visualization
with tabs[1]:
    st.header("PCA Data Visualization")
    if pca_df_train is not None:
        fig = plot_train_dataset(pca_df_train)
        st.pyplot(fig)
    else:
        st.error("No data available for PCA plot.")

# Tab for explained variance ratio
with tabs[2]:
    st.header("Explained Variance Ratio")
    if pca is not None:
        fig = show_variance_ratio(pca)
        st.pyplot(fig)
    else:
        st.error("No PCA model available.")

# Tab for model evaluation
with tabs[3]:
    st.header("Model Evaluation")

    # Add noise option
    if st.checkbox("Add noise to the data"):
        X_train_pca = add_noise(0, 0.1, X_train_pca)

    models = {
        "Logistic Regression": LR_accuracy,
        "K-Nearest Neighbors": KNN_accuracy,
        "Naive Bayes": NB_accuracy,
        "Random Forest": RF_accuracy,
        "XGBoost": XGB_accuracy,
        "LDA": LDA_accuracy
    }

    selected_model = st.selectbox("Select a model", list(models.keys()))

    if st.button("Evaluate Model"):
        train_time, cv_time, test_time, cv_scores, avg_cv_score = models[selected_model](X_train_pca, y_train, X_test_pca)

        st.write(f"**Training Time:** {train_time:.4f} seconds")
        st.write(f"**Cross-validation Time:** {cv_time:.4f} seconds")
        st.write(f"**Testing Time:** {test_time:.4f} seconds")
        st.write(f"**Cross-validation Scores:** {cv_scores}")
        st.write(f"**Average Cross-validation Score:** {avg_cv_score:.4f}")
