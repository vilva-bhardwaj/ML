# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
#1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)
    
def training(clf, epocs, device):
    # Instance of the neural network, loss, optimizer 
    opt = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss() 
    for epoch in range(epocs): # train for 10 epochs
        for batch in dataset: 
            x, y = batch 
            x, y = x.to(device), y.to(device) 
            yhat = clf(x) 
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            opt.zero_grad()
            loss.backward() 
            opt.step() 
        print(f"Epoch:{epoch} loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

def inference(clf, img, device):
    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open(img) 
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    print(torch.argmax(clf(img_tensor)))

# Training flow 
if __name__ == "__main__": 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = ImageClassifier().to(device)
    # for training
    # training(clf, 10, device)

    # for inference
    inference(clf, 'img_3.jpg', device)

    inference(clf, 'img_2.jpg', device)

    inference(clf, 'img_1.jpg', device)
