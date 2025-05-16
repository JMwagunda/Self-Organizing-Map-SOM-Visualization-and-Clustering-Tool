# 🧠 Self-Organizing Map (SOM) Visualization and Clustering Tool

This project implements a fully functional Self-Organizing Map (SOM) from scratch using Python and provides interactive visualization tools to explore and understand high-dimensional data.

## 🚀 Features

- ✅ Custom-built SOM class (no external SOM libraries)
- ✅ Exponential decay function for sigma and learning rate
- ✅ Training with customizable iterations & verbose mode
- ✅ Visualization tools:
  - **U-Matrix (Unified Distance Matrix)**
  - **Component Planes Visualization**
  - **Cluster Map for labeled data**
  - **Hit Map (Data frequency map)**
  - **3D Weight Visualization in Feature Space (Plotly)**
- ✅ Batch prediction of BMUs (Best Matching Units)
- ✅ Flexible for multiple datasets (Iris, Wine, Breast Cancer, synthetic blobs, moons, circles)
- ✅ Integration-ready with Streamlit for UI

## 🛠️ Tech Stack

- **Python** (Core implementation)
- **NumPy** (Matrix & vector operations)
- **Pandas** (Data handling)
- **Matplotlib** (2D Visualizations)
- **Plotly** (Interactive 3D visualizations)
- **scikit-learn** (Datasets & preprocessing)
- **Streamlit** (Optional interactive dashboard)
- **PIL** (Image processing utilities)

## 📊 How It Works

1. **Initialization:**
   - SOM grid is initialized with random weights.
   - Parameters like sigma & learning rate are set.

2. **Training:**
   - Iteratively updates neuron weights based on input samples.
   - Uses a neighborhood function (Gaussian) to adjust surrounding neurons.
   - Exponential decay reduces sigma & learning rate over time.

3. **Visualization:**
   - Various plots to analyze how data maps onto the SOM.
   - Unified distance matrix for cluster boundary detection.
   - Component planes for feature-wise interpretation.
   - Hit Maps to visualize data density.
   - Interactive 3D weight grids using Plotly.

## 📦 Folder Structure (Suggested)

