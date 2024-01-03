import torch
from torch.utils.data import TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys

def visualize(model, images, labels) -> None:
    """Visualize the model's predictions on images."""
    print("Visualizing")

    print(model)
    model = torch.load(model).to("cpu")

    images = torch.load(images)
    labels = torch.load(labels)

    print('loaded components')

    predictset = TensorDataset(images, labels)
    predictloader = torch.utils.data.DataLoader(predictset, batch_size=64, shuffle=True)

    feature_set = []
    labels_set = []
    with torch.no_grad():
        for images, labels in predictloader:
            features = model.extract_features(images)
            feature_set.append(features)
            labels_set.append(labels)

    feature_set = torch.cat(feature_set, dim=0).numpy()
    labels_set = torch.cat(labels_set, dim=0).numpy()


    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(feature_set)

    # Visualize the 2D features
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels_set, cmap='viridis', alpha=0.5)
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar()
    plt.savefig('reports/figures/tsne.png')

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python your_script.py <model_path> <images_path> <labels_path>")
    else:
        model_path = sys.argv[1]
        images_path = sys.argv[2]
        labels_path = sys.argv[3]
        visualize(model_path, images_path, labels_path)