import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def visualize_u_matrix(som, figsize=(10, 8)):
    """Visualize the U-Matrix"""
    fig, ax = plt.subplots(figsize=figsize)
    u_matrix = som.get_u_matrix()
    im = ax.imshow(u_matrix, cmap='viridis')
    ax.set_title('U-Matrix (Unified Distance Matrix)')
    plt.colorbar(im)
    return fig

def visualize_component_planes(som, feature_names=None, figsize=(15, 10)):
    """Visualize component planes for each feature"""
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(som.input_dim)]

    cols = min(3, som.input_dim)
    rows = (som.input_dim + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1 and cols == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for i in range(som.input_dim):
        if i < len(axes):
            component_plane = som.weights[:, :, i]
            im = axes[i].imshow(component_plane, cmap='viridis')
            axes[i].set_title(f'{feature_names[i]}')
            plt.colorbar(im, ax=axes[i])

    # Hide unused subplots
    for i in range(som.input_dim, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig

def visualize_cluster_map(som, data, labels, figsize=(10, 8)):
    """Visualize clusters on the map"""
    _, label_map = som.get_map_representation(data, labels)

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

def visualize_hit_map(som, data, figsize=(10, 8)):
    """Visualize hit map (where samples land on the map)"""
    som_map = som.get_map_representation(data)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(som_map, cmap='viridis')
    ax.set_title('SOM Hit Map')
    plt.colorbar(im)

    return fig

def visualize_3d_weights(som, feature_indices=(0, 1, 2), feature_names=None):
    """Visualize 3 weight dimensions in 3D space"""
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(som.input_dim)]

    # Extract weights for selected features
    x_index, y_index, z_index = feature_indices

    # Create a 3D plotly figure
    fig = go.Figure()

    for i in range(som.x):
        for j in range(som.y):
            # Add a point for each neuron
            fig.add_trace(go.Scatter3d(
                x=[som.weights[i, j, x_index]],
                y=[som.weights[i, j, y_index]],
                z=[som.weights[i, j, z_index]],
                mode='markers',
                marker=dict(
                    size=5,
                    color=f'rgb({int(255*i/som.x)}, {int(255*j/som.y)}, 100)',
                    opacity=0.8
                ),
                text=f"Neuron ({i}, {j})"
            ))

            # Add lines to connect neighboring neurons
            if i < som.x - 1:
                fig.add_trace(go.Scatter3d(
                    x=[som.weights[i, j, x_index], som.weights[i+1, j, x_index]],
                    y=[som.weights[i, j, y_index], som.weights[i+1, j, y_index]],
                    z=[som.weights[i, j, z_index], som.weights[i+1, j, z_index]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))

            if j < som.y - 1:
                fig.add_trace(go.Scatter3d(
                    x=[som.weights[i, j, x_index], som.weights[i, j+1, x_index]],
                    y=[som.weights[i, j, y_index], som.weights[i, j+1, y_index]],
                    z=[som.weights[i, j, z_index], som.weights[i, j+1, z_index]],
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

def visualize_data_projection(som, data, labels, map_size_x, map_size_y, feature_names=None):
    """Visualize data projection on the SOM"""
    # Get BMUs for each data point
    bmus = som.predict_batch(data)

    # Create a DataFrame for the visualization
    df_projection = pd.DataFrame({
        'BMU_x': bmus[:, 0],
        'BMU_y': bmus[:, 1],
        'Class': labels
    })

    # Add jitter to avoid overlapping points
    jitter_amount = 0.2
    df_projection['BMU_x_jitter'] = df_projection['BMU_x'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df_projection))
    df_projection['BMU_y_jitter'] = df_projection['BMU_y'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df_projection))

    # Create a scatter plot using Plotly
    fig_proj = px.scatter(
        df_projection,
        x='BMU_x_jitter',
        y='BMU_y_jitter',
        color='Class',
        title="Data Projection on SOM Grid",
        labels={'BMU_x_jitter': 'SOM X coordinate', 'BMU_y_jitter': 'SOM Y coordinate'},
        hover_data=['BMU_x', 'BMU_y', 'Class']
    )

    # Update layout to match the SOM grid dimensions
    fig_proj.update_layout(
        xaxis=dict(range=[-0.5, map_size_x - 0.5]),
        yaxis=dict(range=[-0.5, map_size_y - 0.5]),
        xaxis_title="SOM X coordinate",
        yaxis_title="SOM Y coordinate"
    )

    # Add grid lines to represent the SOM grid
    for i in range(map_size_x + 1):
        fig_proj.add_shape(
            type="line",
            x0=i-0.5, y0=-0.5,
            x1=i-0.5, y1=map_size_y-0.5,
            line=dict(color="gray", width=1, dash="dash")
        )

    for j in range(map_size_y + 1):
        fig_proj.add_shape(
            type="line",
            x0=-0.5, y0=j-0.5,
            x1=map_size_x-0.5, y1=j-0.5,
            line=dict(color="gray", width=1, dash="dash")
        )

    return fig_proj