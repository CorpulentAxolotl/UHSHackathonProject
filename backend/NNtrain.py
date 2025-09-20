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

def train_loop(model, CrossEntrop_loss_fn, optimizer, train_data, epochs): # call in dataloader, loss_fn, optimizer; from torch.utils.data
    print(f"Training...")

    # set model to training mode best practice
    model.train() # my model NN object instantiated from NN blueprint (subclass of nn.Module)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

## for each epoch, run all batches (avoids extent of overfitting)
    
    for epoch in range(epochs):
        batch_accs = []  
        batch_loss = []  
        for batch_num, (x, Y) in enumerate(train_dataloader): # each batch is trained

            # each batch is fed forward and backward in propogation
            logits = model.forward(x) # forward method from NN class can be called with instance of NN class which is model object created from NN class; self can be ignored because we are operating on an instance of NN already, self only to initiate
            y = model.classify(logits) # x is a batch of 64 tensor images
            avg_loss = CrossEntrop_loss_fn(logits, Y) # cross entp expects raw logits and target classes - scalars
            
            # backprop
            avg_loss.backward() # .backward() computes negative gradient of batch average loss; only computes/accumulates grads
            # update and optimize w's, b's
            optimizer.step()
            optimizer.zero_grad()

# compute and print an average loss per epoch

            # print(f"Batch {batch_num+1}  Epoch: {epoch}  Loss: {avg_loss}") 
            batch_acc = batch_accuracy(y, Y) # let accuracy metrics print accuracy
            batch_accs.append(batch_acc)

            batch_loss.append(avg_loss)

            # print accuracy metrics per batch using accuracy metric function
        epoch_acc = sum(batch_accs) / len(batch_accs)
        epoch_loss = sum(batch_loss) / len(batch_loss)
        print(f"Epoch {epoch} Accuracy: {epoch_acc} Loss: {epoch_loss}")

    return model # trained model

def test_loop(model, CrossEntrop_loss_fn):
    model.eval()
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    accuracy_list = []
    loss_list = []
    print(f"Testing...")
    with torch.no_grad(): # stops computing gradients
        for batch_num, (X, Y) in enumerate(test_dataloader): # x and y are tensors of 64 elements; an element of the x tensor is a 28x28 tensor image
            logits = model(X) # input data fed in batches of 64 (64 image tensors)
            y = model.classify(logits)
            # output is a list of predictions of one batch
            avg_loss = CrossEntrop_loss_fn(logits, Y)

            accuracy = batch_accuracy(y, Y) # name the function output as the batch accuracy
            accuracy_list.append(accuracy)
            loss_list.append(avg_loss)
            if batch_num==20: # stop at 50 batches
                break
    
    avg_accuracy = sum(accuracy_list)/len(accuracy_list)
    fin_loss = sum(loss_list) / len(loss_list)
    print(f"Avg accuracy: {avg_accuracy:.2f} Avg loss: {fin_loss}")

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
    test_data = datasets.FashionMNIST(
        root = "data", 
        train=False, 
        download=True, 
        transform=ToTensor()
    )

    print("Creating model, loss function, and optimizer...")
    my_model = NeuralNetwork() # instantiate my neural network
    my_loss_fn = torch.nn.CrossEntropyLoss() ## already has a softmax operation
    my_optimizer = torch.optim.SGD(my_model.parameters(), lr=0.01) # .params accesses all the weights and biases
    
    num_epochs = int(input(("Epochs: ")))

    trained_model = train_loop(my_model, my_loss_fn, my_optimizer, train_data, num_epochs) # idk if this is right
    torch.save(trained_model.state_dict(), "model.pth")