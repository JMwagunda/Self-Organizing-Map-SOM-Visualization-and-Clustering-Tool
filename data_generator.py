import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles

def generate_clustered_data(n_samples=1000, n_features=4, n_clusters=5, random_state=42):
    """
    Generate a clustered dataset with multiple features using make_blobs
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.8,
        random_state=random_state
    )

    # Create a DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = y

    return df

def generate_mixed_geometry_data(n_samples=1000, random_state=42):
    """
    Generate a dataset with various geometric patterns (circles, moons)
    combined with standard clusters
    """
    # Generate two moon-shaped clusters
    X_moons, y_moons = make_moons(n_samples=n_samples//3, noise=0.1, random_state=random_state)

    # Generate concentric circles
    X_circles, y_circles = make_circles(n_samples=n_samples//3, noise=0.05, factor=0.5, random_state=random_state)

    # Generate standard blobs for the remaining features
    X_blobs, y_blobs = make_blobs(
        n_samples=n_samples//3,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        random_state=random_state
    )

    # Combine the datasets
    X = np.vstack([X_moons, X_circles, X_blobs])

    # Adjust the labels to be consecutive
    y = np.concatenate([
        y_moons,
        y_circles + 2,
        y_blobs + 4
    ])

    # Add two more features with some correlation to the clusters
    additional_features = np.zeros((X.shape[0], 2))

    for i in range(7):  # 7 total clusters
        mask = (y == i)
        # Create feature values that correlate with the cluster
        additional_features[mask, 0] = np.random.normal(i, 0.5, size=np.sum(mask))
        additional_features[mask, 1] = np.random.normal(i % 3, 0.7, size=np.sum(mask))

    # Combine all features
    X_final = np.hstack([X, additional_features])

    # Create DataFrame
    feature_names = ['x', 'y', 'feature_3', 'feature_4']
    df = pd.DataFrame(X_final, columns=feature_names)
    df['cluster'] = y

    return df

def generate_temporal_data(n_samples=1000, n_features=4, n_patterns=3, random_state=42):
    """
    Generate temporal data with cyclical patterns
    """
    np.random.seed(random_state)

    # Time points
    t = np.linspace(0, 10, n_samples)

    # Initialize data matrix
    X = np.zeros((n_samples, n_features))

    # Pattern types
    patterns = []

    # Create different temporal patterns
    for i in range(n_patterns):
        frequency = 0.5 + i * 0.5  # Different frequencies
        phase = np.random.uniform(0, 2*np.pi)  # Random phase
        amplitude = 1 + np.random.uniform(-0.5, 0.5)  # Slightly varying amplitude

        pattern = amplitude * np.sin(frequency * t + phase)
        patterns.append(pattern)

    # Create features with combinations of patterns and noise
    for j in range(n_features):
        base_pattern = patterns[j % n_patterns]
        noise_level = 0.2 + 0.1 * j

        # Add some pattern mixing and noise
        X[:, j] = base_pattern + noise_level * np.random.randn(n_samples)

        # Add some non-linearity
        if j % 2 == 0:
            X[:, j] = X[:, j] + 0.5 * X[:, j]**2

    # Create clusters based on the dominant pattern
    y = np.zeros(n_samples)
    segment_size = n_samples // n_patterns

    for i in range(n_patterns):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_patterns - 1 else n_samples
        y[start_idx:end_idx] = i

    # Create a DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['time'] = t
    df['pattern'] = y

    return df

def generate_image_data(image_size=28, n_samples=500, n_shapes=3, random_state=42):
    """
    Generate synthetic image-like data with different shapes
    """
    np.random.seed(random_state)

    # Initialize data array
    X = np.zeros((n_samples, image_size * image_size))
    y = np.zeros(n_samples)

    samples_per_class = n_samples // n_shapes

    for shape_idx in range(n_shapes):
        start_idx = shape_idx * samples_per_class
        end_idx = (shape_idx + 1) * samples_per_class if shape_idx < n_shapes - 1 else n_samples

        for i in range(start_idx, end_idx):
            # Create a blank image
            img = np.zeros((image_size, image_size))

            # Position variations
            center_x = np.random.randint(image_size // 4, 3 * image_size // 4)
            center_y = np.random.randint(image_size // 4, 3 * image_size // 4)

            # Size variations
            size = np.random.randint(image_size // 6, image_size // 3)

            # Intensity variation
            intensity = 0.5 + np.random.random() * 0.5

            # Draw different shapes
            if shape_idx == 0:  # Squares
                x_min = max(0, center_x - size // 2)
                x_max = min(image_size, center_x + size // 2)
                y_min = max(0, center_y - size // 2)
                y_max = min(image_size, center_y + size // 2)

                img[y_min:y_max, x_min:x_max] = intensity

            elif shape_idx == 1:  # Circles
                y_grid, x_grid = np.ogrid[:image_size, :image_size]
                dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                mask = dist <= size // 2
                img[mask] = intensity

            elif shape_idx == 2:  # Crosses
                thickness = max(1, size // 5)

                # Horizontal line
                y_min = max(0, center_y - thickness // 2)
                y_max = min(image_size, center_y + thickness // 2)
                x_min = max(0, center_x - size // 2)
                x_max = min(image_size, center_x + size // 2)
                img[y_min:y_max, x_min:x_max] = intensity

                # Vertical line
                y_min = max(0, center_y - size // 2)
                y_max = min(image_size, center_y + size // 2)
                x_min = max(0, center_x - thickness // 2)
                x_max = min(image_size, center_x + thickness // 2)
                img[y_min:y_max, x_min:x_max] = intensity

            # Add some noise
            img += np.random.normal(0, 0.05, (image_size, image_size))
            img = np.clip(img, 0, 1)

            # Flatten and store
            X[i] = img.flatten()
            y[i] = shape_idx

    # Create a DataFrame with flattened pixels
    pixel_columns = [f'pixel_{i}' for i in range(image_size * image_size)]
    df = pd.DataFrame(X, columns=pixel_columns)
    df['shape'] = y

    return df

def generate_customer_segmentation_data(n_samples=1000, random_state=42):
    """
    Generate a synthetic customer segmentation dataset
    """
    np.random.seed(random_state)

    # Initialize data
    data = {
        'age': np.zeros(n_samples),
        'income': np.zeros(n_samples),
        'spending_score': np.zeros(n_samples),
        'purchase_frequency': np.zeros(n_samples),
        'loyalty_years': np.zeros(n_samples)
    }

    # Define segments and their characteristics
    segments = {
        0: {  # Young budget shoppers
            'age': (20, 30),
            'income': (20000, 40000),
            'spending_score': (20, 50),
            'purchase_frequency': (1, 5),
            'loyalty_years': (0, 2)
        },
        1: {  # Middle-aged mid-market
            'age': (30, 45),
            'income': (40000, 70000),
            'spending_score': (40, 70),
            'purchase_frequency': (4, 10),
            'loyalty_years': (1, 5)
        },
        2: {  # Affluent shoppers
            'age': (35, 60),
            'income': (70000, 150000),
            'spending_score': (70, 100),
            'purchase_frequency': (8, 20),
            'loyalty_years': (3, 10)
        },
        3: {  # Senior conservative
            'age': (55, 80),
            'income': (30000, 60000),
            'spending_score': (30, 60),
            'purchase_frequency': (2, 6),
            'loyalty_years': (5, 15)
        },
        4: {  # High-value loyalists
            'age': (40, 65),
            'income': (80000, 200000),
            'spending_score': (80, 100),
            'purchase_frequency': (15, 30),
            'loyalty_years': (7, 20)
        }
    }

    # Generate segment assignments
    segment_probs = [0.25, 0.30, 0.20, 0.15, 0.10]  # Probability of each segment
    y = np.random.choice(list(segments.keys()), size=n_samples, p=segment_probs)

    # Generate feature values based on segment
    for i in range(n_samples):
        segment = segments[y[i]]

        for feature, (min_val, max_val) in segment.items():
            # Generate value from a truncated normal distribution within the range
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 4

            # Generate a value and ensure it's within the range
            value = np.random.normal(mean, std)
            value = max(min_val, min(max_val, value))

            data[feature][i] = value

    # Add some correlations and non-linearities

    # Spending score correlates with income (with some randomness)
    for i in range(n_samples):
        data['spending_score'][i] += (data['income'][i] / 200000) * 20 * np.random.uniform(0.7, 1.3)
        data['spending_score'][i] = min(100, max(0, data['spending_score'][i]))

    # Purchase frequency correlates with spending score and has some non-linearity
    for i in range(n_samples):
        data['purchase_frequency'][i] += (data['spending_score'][i] / 100) * 10 * np.random.uniform(0.8, 1.2)
        # Add non-linearity
        if data['age'][i] < 30:
            data['purchase_frequency'][i] *= 1.2  # Young people shop more frequently

        data['purchase_frequency'][i] = max(1, data['purchase_frequency'][i])

    # Create DataFrame
    df = pd.DataFrame(data)

    # Scale income to be in thousands for better visualization
    df['income'] = df['income'] / 1000

    # Add segment labels
    df['segment'] = y

    return df

# Note: The save_datasets and visualize_datasets functions are primarily for generating
# and visualizing the synthetic data outside of the Streamlit app.
# You might not need them directly in the Streamlit flow, but you can keep them
# here if you want to be able to generate and save the datasets separately.
# If you only need to generate data for the Streamlit app, you can omit these.
# from data_generator import save_datasets, visualize_datasets # Example if you keep them