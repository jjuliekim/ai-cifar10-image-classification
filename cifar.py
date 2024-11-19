import random
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt

# load the CIFAR-10 data set 
training_data = datasets.CIFAR10(
  root="data",
  train=True,
  download=True,
  transform=ToTensor(),
)

test_data = datasets.CIFAR10(
  root="data",
  train=False,
  download=True,
  transform=ToTensor(),
)

# split into batches and create data loaders
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
device = "cpu"

# model 1: feed-forward network with 2 hidden layers
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(32 * 32 * 3, 1024),
      nn.BatchNorm1d(1024), # using batch normalization
      nn.ReLU(), # using ReLU activation function
      nn.Linear(1024, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, 10)
    )
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
  
# model 2: convolutional network with 2 hidden convolutional layers
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2), # applying max pooling
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(), # flatten before Linear layer
      nn.Linear(128 * 8 * 8, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Linear(256, 10)
    )
  def forward(self, x):
    logits = self.conv_layers(x)
    return logits

feed_forward_model = NeuralNetwork().to(device)
cnn_model = CNN().to(device)

# specify optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
ffnn_optimizer = torch.optim.SGD(feed_forward_model.parameters(), lr=0.001)
cnn_optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.001)

# model training
def train(dataloader, model, loss_fn, optimizer):
  model.train()
  for X, y in dataloader:
    X, y = X.to(device), y.to(device)
    # compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)
    # backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
# track loss for graphing
ffnn_losses = []
cnn_losses = []
      
# model testing
def test(dataloader, model, loss_fn, test_losses):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  # compute accuracy
  test_loss /= num_batches
  test_losses.append(test_loss)
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# training iterations
epochs = 5
for t in range(epochs):
  print(f"=== MODEL 1 - EPOCH {t+1} ===")
  train(train_dataloader, feed_forward_model, loss_fn, ffnn_optimizer)
  test(test_dataloader, feed_forward_model, loss_fn, ffnn_losses)
# save the model
torch.save(feed_forward_model.state_dict(), "feed_forward_model.pth") 

for t in range(epochs):
  print(f"=== MODEL 2 - EPOCH {t+1} ===")
  train(train_dataloader, cnn_model, loss_fn, cnn_optimizer)
  test(test_dataloader, cnn_model, loss_fn, cnn_losses)
torch.save(cnn_model.state_dict(), "cnn_model.pth")

# loss graph model 1
plt.plot(range(1, epochs + 1), ffnn_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model 1 - Loss after each epoch')
plt.grid(True)
plt.show()

# loss graph model 2
plt.plot(range(1, epochs + 1), cnn_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model 2 - Loss after each epoch')
plt.grid(True)
plt.show()

# find image examples from saved models
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
feed_forward_model = NeuralNetwork().to(device)
feed_forward_model.load_state_dict(torch.load("feed_forward_model.pth", weights_only=True))
feed_forward_model.eval()
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load("cnn_model.pth", weights_only=True))
cnn_model.eval()

def find_image_examples(model):
  found_correct = False
  found_incorrect = False
  indices = list(range(len(test_data)))
  random.shuffle(indices)
  
  for i in indices:
    x, y = test_data[i]
    # x = image tensor, y = label
    x = x.to(device)
    with torch.no_grad():
      pred = model(x.unsqueeze(0)) 
      predicted_class = classes[pred.argmax(1).item()]
      actual_class = classes[y]
    # correct example not found yet
    if predicted_class == actual_class and not found_correct:
      plt.imshow(x.permute(1, 2, 0))
      plt.title(f"Correctly Predicted: '{predicted_class}', Actual: '{actual_class}'")
      plt.show()
      found_correct = True
    # incorrect example not found yet
    if predicted_class != actual_class and not found_incorrect:
      plt.imshow(x.permute(1, 2, 0))
      plt.title(f"Incorrectly Predicted: '{predicted_class}', Actual: '{actual_class}'")
      plt.show()
      found_incorrect = True
    # both examples found
    if found_correct and found_incorrect:
      break
    
find_image_examples(feed_forward_model)
find_image_examples(cnn_model)
