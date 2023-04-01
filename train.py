import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader 
from torchvision import transforms 
from dataset import SatelliteImagesDataset 
from model import Inception_ResNet_v2 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

traf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299,299)),
    ])
ds = SatelliteImagesDataset('../satelliteimgs', './filepaths.csv', transform=traf)

dataloader = DataLoader(ds, batch_size=32, shuffle=True)

inception_res_v2 = Inception_ResNet_v2(3, 4).to(device)
inception_res_v2.classifier[4] = nn.Identity() 
optim = torch.optim.RMSprop(inception_res_v2.parameters(), lr=0.045,eps=1.0, weight_decay=0.9)
criterion = nn.CrossEntropyLoss().to(device)

LOAD = False
EPOCHS = 15 
FILENAME=''
# {'state_dict':inception_res_v2.state_dict(), 
#  'optim_state_dict':optim.state_dict()}
def save_checkpoint(checkpoint, filename):
    torch.save(checkpoint, filename)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    inception_res_v2.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])

if LOAD:
    load_checkpoint(FILENAME)
for epoch in range(EPOCHS):
    loop = tqdm(dataloader)
    checkpoint = {
            'state_dict':inception_res_v2.state_dict(),
            'optim_state_dict': optim.state_dict()
            }
    if epoch!=0:
        save_checkpoint(checkpoint, f'checkpoint_{epoch}.pth')
    loop.set_description(f'Epoch[{epoch+1}/{EPOCHS}]')
    for data, label in loop:
        data = data.to(device)
        label = label.to(device)
        pred = inception_res_v2(data) 
        loss = criterion(pred, label) 
        optim.zero_grad()
        loss.backward()
        optim.step()
        loop.set_postfix_str(f'Loss:{loss.item()}')
         
