# validatefid.py

import torch
from torch.utils.data import DataLoader
from fidcnn import get_feature_extractor
from dataset import MinecraftDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def extract_features(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    features = []
    with torch.no_grad():  # No need to track gradients
        for data in dataloader:
            output = model(data)
            features.append(output.cpu().numpy())
    return np.vstack(features)  # Combine all feature arrays

def evaluate_clustering(features):

    # Perform KMeans clustering
    n_clusters = min(10, len(features)) 
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(features)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(features, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

def main():
    data = MinecraftDataset()
    dataloader = DataLoader(data, batch_size=4, shuffle=True)

    model = get_feature_extractor()
    model.load_state_dict(torch.load('fid_3dcnn.pth'))  # Load the trained model

    features = extract_features(model, dataloader)
    evaluate_clustering(features)

if __name__ == "__main__":
    main()
