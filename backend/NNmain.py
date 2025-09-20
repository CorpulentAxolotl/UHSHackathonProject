### Classifying the MNIST Fashion Dataset with Pytorch NN

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer

## create dataset object
## since MNIST dataset is built-in in torchvision, no need for custom class for preprocessing
# datasets.FashionMNIST returns a dataset that can be iterated over when processed into dataloader

## feed dataset object into Dataloader
# DataLoader is a utility class that accepts a Dataset and splits into shuffled batches
# dataloader wraps iterable around data which is data, label -> image tensor, one hot encoded label tensor

## Build the NN
class NeuralNetwork(nn.Module): # Neural Network is subclassed nn.Module
    def __init__(self): # constructor
        super().__init__() # Inherit all class variables from nn.Module
        # construct the linear-relu layers
        self.flatten = nn.Flatten() # instance attribute local to NN constructor
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # number of input features 28*28; 512 output from first neuron layer
            nn.ReLU(), # connections between nodes have individ weights and biases
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
 
    # forward feed the data
    def forward(self, x): ## only forward is called when model(x) because of nn.Module's overriding and inheritance structure
        x = self.flatten(x) # variable local to forward method; x is input data called later on during NN implementation
        logits = self.linear_relu_stack(x)
        
        return logits
    
# compute logits to probability and then to classifications
    def classify(self, logits):
        return logits.argmax(dim=1) # scalar of highest raw logit

def batch_accuracy(y, Y,): #accuracy_list
    correct = (y == Y).sum().item() # boolean tensors where pred_y == Y is True then summed up; tensor to tensor comparison in a batch of 64 tensors
    total = len(Y)
    accuracy = (correct / total)
    
    return accuracy

if __name__ == "__baseNN__":

    train_data = datasets.FashionMNIST(
        root = "data",
        train=True, 
        download=True, 
        transform=ToTensor(),
        target_transform=None
    )

## datasets.FashionMNIST is a subclass of Torch Vision's subclass - datasets

# Test data
    # test_data = datasets.FashionMNIST(
    #     root = "data", 
    #     train=False, 
    #     download=True, 
    #     transform=ToTensor()
    # )

    # data format - tensors
    ...

    print("Creating model, loss function, and optimizer...")
    my_model = NeuralNetwork() # instantiate my neural network
    my_loss_fn = torch.nn.CrossEntropyLoss() ## already has a softmax operation
    my_optimizer = torch.optim.SGD(my_model.parameters(), lr=0.01) # .params accesses all the weights and biases
    
    num_epochs = int(input(("Epochs: ")))

    trained_model = torch.load('model.pth')
    test_loop(trained_model.load_state_dict(), my_loss_fn)