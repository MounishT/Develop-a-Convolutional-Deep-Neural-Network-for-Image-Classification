# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Develop a CNN (Convolutional Neural Network) model using PyTorch to classify images from the Fashion-MNIST dataset into 10 clothing categories such as shirt, shoe, bag, and dress

## Neural Network Model:
<img width="1896" height="777" alt="image" src="https://github.com/user-attachments/assets/f1c3b1a9-8f60-4ecf-ae08-a0a2ad8e93a5" />


## DESIGN STEPS
### STEP 1: 
Load and preprocess the Fashion-MNIST dataset.

### STEP 2: 
Build the CNN architecture with convolution, pooling, and fully connected layers.


### STEP 3: 

Initialize the model, loss function, and optimizer.

### STEP 4: 

Train the CNN model using training data.

### STEP 5: 
Test and evaluate the model performance using accuracy and confusion matrix.


### STEP 6: 

Predict and display the class of a single test image.



## PROGRAM

### Name:ARCHANA T

### Register Number:212223240013

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),        
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))
# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
        from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name:ARCHANA T')
print('Register Number:212223240013')
summary(model, input_size=(1, 28, 28))
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.001)
## Step 3: Train the Model

def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Name: ARCHANA T')
        print('Register Number: 212223240013')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        # Train the model
        train_model(model, train_loader)
        ## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: ARCHANA T')
    print('Register Number: 212223240013')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:ARCHANA T')
    print('Register Number:212223240013 ')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name:ARCHANA T')
    print('Register Number:212223240013')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
# Evaluate the model
test_model(model, test_loader)
## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: ARCHANA T')
    print('Register Number:212223240013')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
# Example Prediction
predict_image(model, image_index=250, dataset=test_dataset)
```

### OUTPUT

## Training Loss per Epoch
<img width="255" height="150" alt="image" src="https://github.com/user-attachments/assets/eb303caa-1aa4-4177-9517-ecc372239e6b" />


## Confusion Matrix
<img width="702" height="575" alt="image" src="https://github.com/user-attachments/assets/e6de163c-c1c3-4f8e-a134-46a817afe992" />


## Classification Report
<img width="450" height="293" alt="image" src="https://github.com/user-attachments/assets/a7262dfa-4084-4d66-a539-476545f261f7" />


### New Sample Data Prediction
<img width="482" height="425" alt="image" src="https://github.com/user-attachments/assets/1cf09146-efba-45ff-abbb-1cdec2d2e97f" />


## RESULT
Thus the program to develop a Convolutional Deep Neural Network for Image Classification
