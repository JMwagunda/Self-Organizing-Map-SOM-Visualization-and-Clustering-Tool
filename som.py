import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import os

def exponential_decay(initial_sigma, initial_learning_rate, i, max_iter):
    """Exponential decay function for learning rate and sigma"""
    # Calculate decayed sigma and learning rate
    sigma = initial_sigma * np.exp(-i / (max_iter / 10))
    learning_rate = initial_learning_rate * np.exp(-i / (max_iter / 10))
    return sigma, learning_rate

class SelfOrganizingMap:
    def __init__(self, x=10, y=10, input_dim=3, sigma=1.0, learning_rate=0.5, decay_function=None, random_seed=None):
        """
        Initialize a Self Organizing Map

        Parameters:
        -----------
        x, y : int
            Dimensions of the SOM grid
        input_dim : int
            Dimensionality of the input data
        sigma : float
            Initial neighborhood radius
        learning_rate : float
            Initial learning rate
        decay_function : function
            Function that reduces learning_rate and sigma with each iteration
        random_seed : int
            Seed for reproducibility
        """
        if random_seed:
            np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.sigma = sigma
        self.initial_sigma = sigma
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.decay_function = decay_function

        # Initialize weights
        self.weights = np.random.rand(x, y, input_dim)

        # Initialize activation map
        self.activation_map = np.zeros((x, y))

        # Precompute grid coordinates for faster calculations
        self.grid_coordinates = np.array([(i, j) for i in range(x) for j in range(y)])

    def get_winner(self, x):
        """Find the best matching unit (BMU) for a data vector x"""
        # Compute distances between input and all neurons
        distances = np.linalg.norm(self.weights - x, axis=2)
        # Find index of neuron with minimum distance (BMU)
        winner = np.unravel_index(np.argmin(distances), distances.shape)
        return winner

    def update(self, x, winner, iteration, max_iterations):
        """Update weights based on winner"""
        # Update sigma and learning rate
        if self.decay_function:
            self.sigma, self.learning_rate = self.decay_function(
                self.initial_sigma, self.initial_learning_rate, iteration, max_iterations
            )

        # For all neurons, compute influence based on distance to winner
        for i in range(self.x):
            for j in range(self.y):
                # Calculate Euclidean distance on map grid
                dist = np.linalg.norm(np.array([i, j]) - np.array(winner))

                # Calculate neighborhood function (Gaussian)
                influence = np.exp(-(dist**2) / (2 * self.sigma**2))

                # Update weight of the neuron
                self.weights[i, j] += self.learning_rate * influence * (x - self.weights[i, j])

    def train(self, data, iterations=10000, verbose=False):
        """Train the SOM on the input data"""
        errors = []  # Track errors during training
        
        for iteration in range(iterations):
            if verbose and iteration % (iterations // 10) == 0:
                print(f"Training iteration {iteration}/{iterations}")

            # Select a random data point
            x = data[np.random.randint(0, len(data))]

            # Find winner
            winner = self.get_winner(x)

            # Calculate error before update
            error = np.mean([np.linalg.norm(x - self.weights[winner])])
            errors.append(error)

            # Update weights
            self.update(x, winner, iteration, iterations)

        if verbose:
            print("Training complete!")
            
        return errors

    def predict(self, x):
        """Predict the BMU for a new data point"""
        return self.get_winner(x)

    def predict_batch(self, data):
        """Predict BMUs for a batch of data"""
        winners = []
        for x in data:
            winners.append(self.predict(x))
        return np.array(winners)

    def get_map_representation(self, data, labels=None):
        """Get a representation of the data on the map"""
        som_map = np.zeros((self.x, self.y))
        if labels is not None:
            label_map = np.zeros((self.x, self.y)) - 1  # -1 represents no assigned label

        for i, x in enumerate(data):
            winner = self.predict(x)
            som_map[winner] += 1
            if labels is not None:
                label_map[winner] = labels[i]

        if labels is not None:
            return som_map, label_map
        return som_map

    def get_u_matrix(self):
        """Compute the U-Matrix (unified distance matrix)"""
        u_matrix = np.zeros((self.x, self.y))

        for i in range(self.x):
            for j in range(self.y):
                # Find neighbors
                neighbors = []
                if i > 0:
                    neighbors.append(self.weights[i-1, j])
                if i < self.x - 1:
                    neighbors.append(self.weights[i+1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j-1])
                if j < self.y - 1:
                    neighbors.append(self.weights[i, j+1])

                # Calculate average distance to neighbors
                avg_dist = np.mean([np.linalg.norm(self.weights[i, j] - neighbor) for neighbor in neighbors])
                u_matrix[i, j] = avg_dist

        return u_matrix

    def visualize_u_matrix(self, figsize=(10, 8)):
        """Visualize the U-Matrix"""
        fig, ax = plt.subplots(figsize=figsize)
        u_matrix = self.get_u_matrix()
        im = ax.imshow(u_matrix, cmap='viridis')
        ax.set_title('U-Matrix (Unified Distance Matrix)')
        plt.colorbar(im)
        return fig

    def visualize_component_planes(self, feature_names=None, figsize=(15, 10)):
        """Visualize component planes for each feature"""
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.input_dim)]

        cols = min(3, self.input_dim)
        rows = (self.input_dim + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if rows == 1 and cols == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i in range(self.input_dim):
            if i < len(axes):
                component_plane = self.weights[:, :, i]
                im = axes[i].imshow(component_plane, cmap='viridis')
                axes[i].set_title(f'{feature_names[i]}')
                plt.colorbar(im, ax=axes[i])

        # Hide unused subplots
        for i in range(self.input_dim, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def visualize_cluster_map(self, data, labels, figsize=(10, 8)):
        """Visualize clusters on the map"""
        _, label_map = self.get_map_representation(data, labels)

        fig, ax = plt.subplots(figsize=figsize)

        # Create a colormap with a color for each unique label
        n_labels = len(np.unique(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
        cmap = ListedColormap(colors)

        # Plot the label map
        im = ax.imshow(label_map, cmap=cmap, interpolation='none')

        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=np.unique(labels))
        cbar.set_label('Class')

        ax.set_title('SOM Cluster Map')

        return fig

    def visualize_hit_map(self, data, figsize=(10, 8)):
        """Visualize hit map (where samples land on the map)"""
        som_map = self.get_map_representation(data)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(som_map, cmap='viridis')
        ax.set_title('SOM Hit Map')
        plt.colorbar(im)

        return fig

    def visualize_3d_weights(self, feature_indices=(0, 1, 2), feature_names=None):
        """Visualize 3 weight dimensions in 3D space"""
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.input_dim)]

        # Extract weights for selected features
        x_index, y_index, z_index = feature_indices

        # Create a 3D plotly figure
        fig = go.Figure()

        for i in range(self.x):
            for j in range(self.y):
                # Add a point for each neuron
                fig.add_trace(go.Scatter3d(
                    x=[self.weights[i, j, x_index]],
                    y=[self.weights[i, j, y_index]],
                    z=[self.weights[i, j, z_index]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=f'rgb({int(255*i/self.x)}, {int(255*j/self.y)}, 100)',
                        opacity=0.8
                    ),
                    text=f"Neuron ({i}, {j})"
                ))

                # Add lines to connect neighboring neurons
                if i < self.x - 1:
                    fig.add_trace(go.Scatter3d(
                        x=[self.weights[i, j, x_index], self.weights[i+1, j, x_index]],
                        y=[self.weights[i, j, y_index], self.weights[i+1, j, y_index]],
                        z=[self.weights[i, j, z_index], self.weights[i+1, j, z_index]],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False
                    ))

                if j < self.y - 1:
                    fig.add_trace(go.Scatter3d(
                        x=[self.weights[i, j, x_index], self.weights[i, j+1, x_index]],
                        y=[self.weights[i, j, y_index], self.weights[i, j+1, y_index]],
                        z=[self.weights[i, j, z_index], self.weights[i, j+1, z_index]],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False
                    ))

        fig.update_layout(
            title="SOM Weight Grid in 3D Feature Space",
            scene=dict(
                xaxis_title=feature_names[x_index],
                yaxis_title=feature_names[y_index],
                zaxis_title=feature_names[z_index]
            ),
            width=800,
            height=800
        )

        return fig