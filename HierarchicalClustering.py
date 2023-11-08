import os
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import extract_patches_2d
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Set the path to the dataset folder (update this with the actual path after downloading the dataset)
dataset_path = '/Users/ichangmin/Desktop/ComputeModeling/archive'

# Define a function to load and preprocess images
def load_and_preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((64, 64))  # Resize to a fixed size
        image_array = np.array(image)
        patches = extract_patches_2d(image_array, (16, 16), max_patches=16, random_state=42)
        features = patches.mean(axis=(1, 2, 3))
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Extract features from images
features_list = []
for root, dirs, files in os.walk(dataset_path):
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file_name)
            features = load_and_preprocess_image(image_path)
            if features is not None:
                features_list.append(features)

# Ensure there are features to cluster
if not features_list:
    raise ValueError("No features were extracted from images. Check your dataset path and image files.")

# Perform hierarchical clustering
X = np.array(features_list)
clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
clustering.fit(X)

# Plot the dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

# Plot the corresponding dendrogram
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clustering, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.ylabel("similarity")
plt.show()
