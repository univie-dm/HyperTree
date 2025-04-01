from sklearn import metrics
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from torchvision import transforms, datasets
from sklearn.cluster import Birch
from src.models.utils import dendrogram_purity


train_ds = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

# Initialize list to store all data
all_data = []

# Collect all data from train loader
for batch, _ in train_loader:
    all_data.append(batch.view(batch.size(0), -1))  # Flatten the images

# Concatenate all batches
all_data = torch.cat(all_data, dim=0)
all_data = all_data.numpy()

# Calculate pairwise distances using Euclidean distance
pd = pairwise_distances(all_data, metric='euclidean')


Birch(n_clusters=100).fit(pd)
pred_labels = Birch(n_clusters=100).predict(pd)

nmi = metrics.normalized_mutual_info_score(train_ds.targets, pred_labels)
ari = metrics.adjusted_rand_score(train_ds.targets, pred_labels)

print(f"NMI: {nmi}, ARI: {ari}")

print(f"Dendrogram purity (single): {dendrogram_purity(pd, train_ds.targets, linkage_method='single')}")
print(f"Dendrogram purity (complete): {dendrogram_purity(pd, train_ds.targets, linkage_method='complete')}")
print(f"Dendrogram purity (average): {dendrogram_purity(pd, train_ds.targets, linkage_method='average')}")
print(f"Dendrogram purity (ward): {dendrogram_purity(pd, train_ds.targets, linkage_method='ward')}")
