import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

device = torch.device('cpu')

train_dataset = torchvision.datasets.FashionMNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data',train=False,transform=transforms.ToTensor())

isize = 784 #flatten out
hsize = 64 
classes = 10
epochs = 80
batch = 100
lr = 0.01


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch)


class NNTW(nn.Module):
    def __init__(self, input_size,hidden_size,classes):
        super(NNTW,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,classes)

    

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NNTW(isize,hsize,classes).to(device)

cost = nn.CrossEntropyLoss()
train = torch.optim.SGD(model.parameters(),lr = lr)

nts = len(train_loader)
for ee in range(epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1,784).to(device)
        labels = labels.to(device)

        Youts = model(images)
        loss = cost(Youts,labels)

        loss.backward()
        train.step()
        train.zero_grad()

        if (i+1)%100 == 0:
            print(ee+1,"/",epochs ,"step", i+1,'/',nts,": ",loss.item())


with torch.no_grad():
    nc = 0
    ns = len(test_loader.dataset)
    for images,labels in test_loader:
        images = images.reshape(-1,784).to(device)
        labels = labels.to(device)

        outs = model(images)

        _,pred = torch.max(outs,1)
        nc = nc + (pred == labels).sum().item()

    print("Accuracy over",ns ,"is ", 100*nc/ns)
