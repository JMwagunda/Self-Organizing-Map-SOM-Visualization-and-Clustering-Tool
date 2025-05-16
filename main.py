import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Import classes and functions from your other files
from som import SelfOrganizingMap, exponential_decay
from visualization import (
    visualize_u_matrix,
    visualize_component_planes,
    visualize_cluster_map,
    visualize_hit_map,
    visualize_3d_weights,
    visualize_data_projection
)

# Import data generation functions
from data_generator import (
    generate_clustered_data,
    generate_mixed_geometry_data,
    generate_temporal_data,
    generate_image_data,
    generate_customer_segmentation_data
)


def main():
    st.set_page_config(layout="wide", page_title="Self-Organizing Map Visualization")

    st.title("Self-Organizing Map (SOM) Interactive Visualization")

    st.write("""
    ## What is a Self-Organizing Map?

    A Self-Organizing Map (SOM) is a type of artificial neural network that is trained using unsupervised learning
    to produce a low-dimensional (typically two-dimensional) representation of the input space, called a map.
    SOMs are useful for visualizing high-dimensional data in a way that preserves the topological properties of the input space.

    This application allows you to train and visualize a SOM on different datasets.
    """)

    # Dataset selection
    st.sidebar.header("SOM Configuration")
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["Iris", "Wine", "Breast Cancer", "Clustered Synthetic", "Mixed Geometry Synthetic", "Temporal Synthetic", "Image Synthetic (small)", "Customer Segmentation Synthetic", "Custom Upload"]
    )

    # SOM parameters
    st.sidebar.subheader("SOM Parameters")
    map_size_x = st.sidebar.slider("Map Width", 5, 30, 10)
    map_size_y = st.sidebar.slider("Map Height", 5, 30, 10)
    iterations = st.sidebar.slider("Training Iterations", 100, 10000, 1000, step=100)  # Reduced max iterations
    learning_rate = st.sidebar.slider("Initial Learning Rate", 0.01, 1.0, 0.5)
    sigma = st.sidebar.slider("Initial Sigma (Neighborhood Radius)", 0.1, 5.0, 1.0)
    random_seed = st.sidebar.number_input("Random Seed (for reproducibility)", value=42)

    # Load or generate data based on selection
    X = None
    y = None
    feature_names = None
    df = None

    if dataset_name == "Iris":
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
    elif dataset_name == "Wine":
        data = load_wine()
        X = data.data
        y = data.target
        feature_names = data.feature_names
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names
    elif dataset_name == "Clustered Synthetic":
        df = generate_clustered_data()
        X = df.drop('cluster', axis=1).values
        y = df['cluster'].values
        feature_names = df.drop('cluster', axis=1).columns.tolist()
    elif dataset_name == "Mixed Geometry Synthetic":
        df = generate_mixed_geometry_data()
        X = df.drop('cluster', axis=1).values
        y = df['cluster'].values
        feature_names = df.drop('cluster', axis=1).columns.tolist()
    elif dataset_name == "Temporal Synthetic":
        df = generate_temporal_data()
        feature_cols = [col for col in df.columns if col not in ['time', 'pattern']]
        X = df[feature_cols].values
        y = df['pattern'].values
        feature_names = feature_cols
    elif dataset_name == "Image Synthetic (small)":
        df = generate_image_data(image_size=16, n_samples=300)
        feature_cols = [col for col in df.columns if col != 'shape']
        X = df[feature_cols].values
        y = df['shape'].values
        feature_names = feature_cols
    elif dataset_name == "Customer Segmentation Synthetic":
        df = generate_customer_segmentation_data()
        feature_cols = [col for col in df.columns if col != 'segment']
        X = df[feature_cols].values
        y = df['segment'].values
        feature_names = feature_cols
    else:  # Custom Upload
        st.sidebar.subheader("Upload CSV Data")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.write("Data Preview:")
                st.sidebar.write(df.head())

                st.sidebar.subheader("Select Columns")
                feature_cols = st.sidebar.multiselect(
                    "Select Feature Columns",
                    df.columns.tolist(),
                    default=df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:4]
                )

                target_col = st.sidebar.selectbox(
                    "Select Target Column (optional)",
                    ["None"] + df.columns.tolist()
                )

                if feature_cols:
                    X = df[feature_cols].values
                    feature_names = feature_cols
                    if target_col != "None":
                        y = df[target_col].values
                    else:
                        y = np.zeros(len(X))
                else:
                    st.error("Please select at least one feature column")
                    return
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
        else:
            st.info("Please select or upload a dataset.")
            return

    if X is None:
        st.info("Please select or upload a dataset.")
        return

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and train SOM
    col1, col2 = st.columns([1, 3])

    with col1:
        train_button = st.button("Train SOM")

    with col2:
        progress_placeholder = st.empty()
    
    if train_button:
        progress_bar = progress_placeholder.progress(0)
        all_errors = []

        # Initialize SOM
        som = SelfOrganizingMap(
            x=map_size_x,
            y=map_size_y,
            input_dim=X_scaled.shape[1],
            sigma=sigma,
            learning_rate=learning_rate,
            decay_function=exponential_decay,
            random_seed=random_seed
        )

        # Train in smaller batches to update progress
        batch_size = iterations // 10
        for i in range(10):
            batch_errors = som.train(X_scaled, iterations=batch_size, verbose=False)
            all_errors.extend(batch_errors)
            progress_bar.progress((i + 1) * 10)

        progress_placeholder.success("Training Complete!")

        # Add smoothed convergence plot
        st.write("## Training Convergence")
        fig_conv = plt.figure(figsize=(10, 6))
        
        # Calculate moving average for smoothing
        window_size = max(len(all_errors) // 50, 1)  # Adaptive window size
        smoothed_errors = np.convolve(all_errors, np.ones(window_size)/window_size, mode='valid')
        
        # Plot both raw and smoothed errors
        plt.plot(range(len(all_errors)), all_errors, 'lightgray', label='Raw Error', alpha=0.3)
        plt.plot(range(window_size-1, len(all_errors)), smoothed_errors, 'b', label='Smoothed Error')
        
        plt.xlabel('Iteration')
        plt.ylabel('Average Quantization Error')
        plt.title('SOM Training Convergence')
        plt.legend()
        st.pyplot(fig_conv)
        # Display results in tabs
        st.write("## SOM Visualization Results")

        tabs = st.tabs([
            "U-Matrix",
            "Component Planes",
            "Cluster Map",
            "Hit Map",
            "3D Grid Visualization",
            "Data Projection"
        ])

        # Tab 1: U-Matrix
        with tabs[0]:
            st.write("""
            ### U-Matrix
            The U-Matrix visualizes distances between neighboring map units.
            Lighter colors represent larger distances (potential cluster boundaries),
            while darker colors indicate areas where neurons are close together (potential clusters).
            """)
            fig_umatrix = visualize_u_matrix(som, figsize=(10, 8))
            st.pyplot(fig_umatrix)

        # Tab 2: Component Planes
        with tabs[1]:
            st.write("""
            ### Component Planes
            Component planes show the distribution of each input feature across the SOM.
            Each plane represents how a single feature is distributed across the map,
            helping to identify correlations between features.
            """)
            fig_components = visualize_component_planes(som, feature_names=feature_names, figsize=(15, 10))
            st.pyplot(fig_components)

        # Tab 3: Cluster Map
        with tabs[2]:
            st.write("""
            ### Cluster Map
            The cluster map shows how different classes or clusters are distributed across the SOM.
            Each color represents a different class from the original data.
            """)
            fig_clusters = visualize_cluster_map(som, X_scaled, y, figsize=(10, 8))
            st.pyplot(fig_clusters)

        # Tab 4: Hit Map
        with tabs[3]:
            st.write("""
            ### Hit Map
            The hit map shows where samples from the dataset are mapped onto the SOM.
            Brighter areas indicate more samples being mapped to those neurons.
            """)
            fig_hits = visualize_hit_map(som, X_scaled, figsize=(10, 8))
            st.pyplot(fig_hits)

        # Tab 5: 3D Grid Visualization
        with tabs[4]:
            st.write("""
            ### 3D Grid Visualization
            This interactive 3D visualization shows how the SOM grid is positioned in the feature space.
            Each point represents a neuron, and the lines connect neighboring neurons in the grid.
            """)

            if X_scaled.shape[1] > 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_feature = st.selectbox("X-axis feature", options=range(len(feature_names)), 
                                           format_func=lambda i: feature_names[i], key="3d_x")
                with col2:
                    y_feature = st.selectbox("Y-axis feature", options=range(len(feature_names)), 
                                           format_func=lambda i: feature_names[i], 
                                           index=min(1, len(feature_names)-1), key="3d_y")
                with col3:
                    z_feature = st.selectbox("Z-axis feature", options=range(len(feature_names)), 
                                           format_func=lambda i: feature_names[i], 
                                           index=min(2, len(feature_names)-1), key="3d_z")
                feature_indices = (x_feature, y_feature, z_feature)
            else:
                feature_indices = tuple(range(min(3, X_scaled.shape[1])))

            fig_3d = visualize_3d_weights(som, feature_indices=feature_indices, feature_names=feature_names)
            st.plotly_chart(fig_3d, use_container_width=True)

        # Tab 6: Data Projection
        with tabs[5]:
            st.write("""
            ### Data Projection on SOM
            This visualization shows how the original data points are projected onto the SOM.
            Each point represents a data sample, positioned according to its best matching unit (BMU) on the SOM grid.
            """)
            fig_proj = visualize_data_projection(som, X_scaled, y, map_size_x, map_size_y, feature_names)
            st.plotly_chart(fig_proj, use_container_width=True)

            st.write("""
            In this visualization:
            - Each point represents a data sample from the original dataset
            - Points are colored by their class/label
            - The grid represents the SOM's neuron layout
            - Points are positioned at their BMU with a small amount of jitter to avoid overlapping
            - Clusters in the original data should appear as groups of similarly colored points
            """)

        # Add SOM information section
        st.write("## SOM Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**SOM Parameters:**")
            st.write(f"- Grid Size: {map_size_x} × {map_size_y}")
            st.write(f"- Input Dimensions: {X_scaled.shape[1]}")
            st.write(f"- Training Iterations: {iterations}")

        with col2:
            st.write("**Learning Parameters:**")
            st.write(f"- Initial Learning Rate: {learning_rate}")
            st.write(f"- Initial Sigma: {sigma}")
            st.write(f"- Decay Function: Exponential")

        with col3:
            st.write("**Dataset Information:**")
            st.write(f"- Dataset: {dataset_name}")
            st.write(f"- Samples: {X_scaled.shape[0]}")
            st.write(f"- Features: {X_scaled.shape[1]}")
            if y is not None:
                st.write(f"- Classes: {len(np.unique(y))}")

    else:
        st.info("Click 'Train SOM' to start training and see visualizations.")

if __name__ == "__main__":
    main()