# helper.py

"""
This module contains custom dataset classes and utility functions for loading, processing,
and evaluating images of celebrities in different caricature styles.

Classes:
1. CelebCariDataset:
   - Loads training/validation images with person and style labels.
   - Supports image transformations.

2. CelebCariTestDataset:
   - Loads test images without labels.
   - Supports image transformations.

Functions:
1. calculate_accuracy:
   - Computes accuracy and average loss for person and style predictions.

2. save_gallery_to_json:
   - Saves gallery embeddings to a JSON file.
   
3. create_gallery_embeddings:
   - Creates gallery embeddings using a model and dataset.
   
4. read_gallery_from_json:
   - Reads gallery embeddings from a JSON file.
   
5. encode_embedding:
   - Encodes an embedding to a base64 string.
   
6. write_predictions_to_json:
   - Writes predictions to a JSON file.

Usage:
- Import the classes and functions and create dataset instances by specifying the root directory and transformations.
- Use with a DataLoader to iterate through the data.
- Use calculate_accuracy to evaluate model performance.
- Use save_gallery_to_json to save embeddings to a JSON file.
- Use create_gallery_embeddings to generate embeddings from a model and dataset.
- Use read_gallery_from_json to load embeddings from a JSON file.
- Use encode_embedding to encode a tensor to a base64 string.
- Use write_predictions_to_json to save predictions to a JSON file.

Example:
from helper import CelebCariDataset, CelebCariTestDataset, calculate_accuracy, save_gallery_to_json, create_gallery_embeddings, read_gallery_from_json, encode_embedding, write_predictions_to_json
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Training dataset
train_dataset = CelebCariDataset(root_dir='path_to_train_dataset', transform=transform)
train_image, person_label, style_label = train_dataset[0]

# Test dataset
test_dataset = CelebCariTestDataset(root_dir='path_to_test_dataset', transform=transform)
test_image, image_path = test_dataset[0]

# Calculate accuracy
accuracy_person, accuracy_style, avg_loss_person, avg_loss_style = calculate_accuracy(
    predicted_person, predicted_style, person_labels, style_labels, running_loss_person, running_loss_style, total)

# Create and save gallery embeddings
gallery_embeddings = create_gallery_embeddings(model, train_dataset, device)
save_gallery_to_json(gallery_embeddings, 'gallery_embeddings.json')

# Read gallery embeddings from JSON
loaded_gallery_embeddings = read_gallery_from_json('gallery_embeddings.json')

# Encode an embedding
encoded_embedding = encode_embedding(gallery_embeddings['person_name'][0])

# Write predictions to JSON
predictions = [('image1.jpg', torch.tensor([1.0, 2.0, 3.0]), 'style1'), ('image2.jpg', torch.tensor([4.0, 5.0, 6.0]), 'style2')]
write_predictions_to_json(predictions, 'predictions.json')
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import base64
import json
import numpy as np
import torch

class CelebCariDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.person_labels = []
        self.style_labels = []
        self.person_to_index = {}
        self.style_to_index = {}
        self.index_to_person = {}
        self.index_to_style = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        for person_folder in os.listdir(self.root_dir):
            person_path = os.path.join(self.root_dir, person_folder)
            if os.path.isdir(person_path):
                if person_folder not in self.person_to_index:
                    self.person_to_index[person_folder] = len(self.person_to_index)
                    self.index_to_person[self.person_to_index[person_folder]] = person_folder
                for file_name in os.listdir(person_path):
                    if file_name.endswith(".jpg") or file_name.endswith(".png"):
                        style_name = file_name.split('_')[2]
                        if style_name not in self.style_to_index:
                            self.style_to_index[style_name] = len(self.style_to_index)
                            self.index_to_style[self.style_to_index[style_name]] = style_name
                        self.image_paths.append(os.path.join(person_path, file_name))
                        self.person_labels.append(self.person_to_index[person_folder])
                        self.style_labels.append(self.style_to_index[style_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        person_label = self.person_labels[idx]
        style_label = self.style_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, person_label, style_label

class CelebCariTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, file_name) for file_name in os.listdir(root_dir)
                            if file_name.endswith(".jpg") or file_name.endswith(".png")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path

def calculate_accuracy(predicted_person, predicted_style, person_labels, style_labels, running_loss_person, running_loss_style, total):
    """
    Computes accuracy and average loss for person and style predictions.

    Args:
    - predicted_person: Tensor of predicted person labels.
    - predicted_style: Tensor of predicted style labels.
    - person_labels: Tensor of true person labels.
    - style_labels: Tensor of true style labels.
    - running_loss_person: Cumulative loss for person predictions.
    - running_loss_style: Cumulative loss for style predictions.
    - total: Total number of samples.

    Returns:
    - accuracy_person: Accuracy of person predictions.
    - accuracy_style: Accuracy of style predictions.
    - average_loss_person: Average loss for person predictions.
    - average_loss_style: Average loss for style predictions.
    """
    correct_person = (predicted_person == person_labels).sum().item()
    correct_style = (predicted_style == style_labels).sum().item()

    accuracy_person = correct_person / total
    accuracy_style = correct_style / total
    average_loss_person = running_loss_person / total
    average_loss_style = running_loss_style / total

    return accuracy_person, accuracy_style, average_loss_person, average_loss_style

def save_gallery_to_json(gallery_embeddings, file_path):
    """
    Saves gallery embeddings to a JSON file.

    Args:
    - gallery_embeddings: Dictionary of person names to embeddings.
    - file_path: Path to the JSON file.

    Returns:
    None
    """
    gallery_dict = {}
    for person_name, embedding in gallery_embeddings.items():
        gallery_dict[person_name] = [encode_embedding(embed) for embed in embedding]
    with open(file_path, 'w') as f:
        json.dump(gallery_dict, f)

def create_gallery_embeddings(model, dataset, device):
    """
    Creates gallery embeddings using a model and dataset.

    Args:
    - model: The model to use for generating embeddings.
    - dataset: The dataset to generate embeddings from.
    - device: The device to perform computations on.

    Returns:
    - gallery_embeddings: Dictionary of person names to list of embeddings.
    """
    model.eval()
    gallery_embeddings = {}
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    with torch.no_grad():
        for inputs, person_labels, _ in dataloader:
            inputs, person_labels = inputs.to(device), person_labels.to(device)
            embeddings, _, _ = model(inputs)
            for embedding, person_label in zip(embeddings, person_labels):
                person_name = dataset.index_to_person[person_label.item()]
                if person_name not in gallery_embeddings:
                    gallery_embeddings[person_name] = []
                gallery_embeddings[person_name].append(embedding.cpu())

    return gallery_embeddings

def read_gallery_from_json(file_path):
    """
    Reads gallery embeddings from a JSON file.

    Args:
    - file_path: Path to the JSON file.

    Returns:
    - gallery_embeddings: Dictionary of person names to list of embeddings.
    """
    with open(file_path, 'r') as f:
        gallery_dict = json.load(f)

    gallery_embeddings = {}
    for person_name, embedding_list in gallery_dict.items():
        gallery_embeddings[person_name] = [torch.tensor(np.frombuffer(base64.b64decode(embedding_str), dtype=np.float32)) for embedding_str in embedding_list]

    return gallery_embeddings

def encode_embedding(embedding):
    """
    Encodes an embedding to a base64 string.

    Args:
    - embedding: A tensor representing the embedding.

    Returns:
    - A base64 encoded string of the embedding.
    """
    return base64.b64encode(embedding.cpu().numpy().astype(np.float32).tobytes()).decode('utf-8')

def write_predictions_to_json(predictions, output_json_path):
    """
    Writes predictions to a JSON file.

    Args:
    - predictions: List of tuples containing filename, embedding, and style.
    - output_json_path: Path to the output JSON file.

    Returns:
    None
    """
    predictions_json = []
    for filename, embedding, style in predictions:
        embedding_str = encode_embedding(embedding)
        predictions_json.append({
            'filename': filename,
            'embedding': embedding_str,
            'style': style
        })

    with open(output_json_path, 'w') as file:
        json.dump(predictions_json, file, indent=4)
