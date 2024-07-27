import argparse
from torchvision import models
import torch
from torch import nn
import numpy as np
from PIL import Image

def get_model(arch, hidden_units):
    model = None
    classifier = None
    input_to_classifier = 0

    # Loading architecture
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_to_classifier = 1024
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_to_classifier = 25088
    else:
        raise ValueError(f"{arch} is unsupported")

    # Freezing parameters
    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(nn.Linear(input_to_classifier, hidden_units),
                           nn.ReLU(),
                           nn.Linear(hidden_units, hidden_units//2),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden_units//2, 102), # 102 classes
                           nn.LogSoftmax(dim=1))
    
    model.classifier = classifier

    return model

def get_input_args_train():
    # Declare Argument Parser
    parser = argparse.ArgumentParser()

    # Setting arguments
    parser.add_argument('data_dir', type=str, help='Directory containing the training data')

    # Optional Arguments
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121', help='Model architecture (e.g., vgg13, densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    return parser.parse_args()

def get_input_args_pred():
    # Declare Argument Parser
    parser = argparse.ArgumentParser()

    # Setting arguments
    parser.add_argument('file_path', type=str, help='Path leading to the image to be tested')
    parser.add_argument('checkpoint_path', type=str, help='Path leading to the pretrained model\'s checkpoint')

    # Optional Arguments
    parser.add_argument('--gpu', action='store_true', help='Use GPU for testing')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes', default=1)
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names', default='')
    
    return parser.parse_args()

def load_model(path):
    checkpoint = torch.load(path)

    # Load architecture
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)


    # Loading Classifer
    classifier = checkpoint['classifier']
    classifier.load_state_dict(checkpoint['state_dict'])

    # Attaching classifier to base model
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(path):
    image = Image.open(path)
    
     # 1- Resizing to get 256 as shortest side
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int((height / width) * 256)
    else:
        new_height = 256
        new_width = int((width / height) * 256)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # 2- Cropping the center out
    upper = (new_height - 224) / 2
    left = (new_width - 224) / 2
    right = (new_width + 224) / 2
    lower = (new_height + 224) / 2
    image = image.crop((left, upper, right, lower))

    # 3- Convert image to a NumPy array
    np_image = np.array(image)
    np_image = np_image.astype(np.float64)

    # 4- Normalize Image
    np_image /= 255.0 # 0-255 -> 0-1
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # 5- Transpose Image
    np_image = np_image.transpose((2, 0, 1))

    return np_image