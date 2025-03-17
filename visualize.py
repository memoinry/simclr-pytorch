import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import models
import myexman
from utils import utils
import argparse
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# 加载配置参数
def load_args():
    parser = argparse.ArgumentParser(description='Visualize model features using t-SNE')
    parser.add_argument('--problem', default='eval', help='The problem to train')
    parser.add_argument('--eval_only', default=True, type=bool, help='Skips the training step if True')
    parser.add_argument('--iters', default=1, type=int, help='The number of optimizer updates')
    parser.add_argument('--arch', default='linear', help='Model architecture')
    parser.add_argument('--ckpt', default='', help='Optional checkpoint to init the model.')
    parser.add_argument('--encoder_ckpt', default='', help='Path to the encoder checkpoint')
    parser.add_argument('--data', default='cifar', help='Dataset to use')
    parser.add_argument('--test_bs', default=256, type=int)
    parser.add_argument('--precompute_emb_bs', default=-1, type=int,
                        help='If it\'s not equal to -1 embeddings are precomputed and fixed before training with batch size equal to this.')
    parser.add_argument('--finetune', default=False, type=bool, help='Finetunes the encoder if True')
    parser.add_argument('--augmentation', default='RandomResizedCrop', help='')
    parser.add_argument('--scale_lower', default=0.08, type=float, help='The minimum scale factor for RandomResizedCrop')

    args = parser.parse_args()
    return args


# 加载数据集
def load_dataset(args):
    if args.data == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")
    dataloader = DataLoader(dataset, batch_size=args.test_bs, shuffle=False)
    return dataloader


# 加载模型
def load_model(args, device):
    model = models.REGISTERED_MODELS[args.problem](args, device=device)
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
    if args.encoder_ckpt != '':
        encoder_ckpt = torch.load(args.encoder_ckpt, map_location=device)
        model.encoder.load_state_dict(encoder_ckpt['state_dict'])
    model.eval()
    return model


# 提取特征
def extract_features(model, dataloader, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            h = model.encode(images)
            features.append(h.cpu().numpy())
            labels.append(targets.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels


# 使用 t-SNE 进行降维并可视化
def tsne_visualization(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend)
    plt.title('t-SNE Visualization of Encoded Features')
    plt.savefig('tsne_visualization.png')
    plt.show()


def main():
    args = load_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = load_dataset(args)
    model = load_model(args, device)

    features, labels = extract_features(model, dataloader, device)
    tsne_visualization(features, labels)


if __name__ == "__main__":
    main()