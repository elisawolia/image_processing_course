import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class DatasetFashion(Dataset):
    def __init__(self, data, transform=None):        
        self.fashion_data = list(data.values)
        self.transform = transform

        label, image = [], []
        for i in self.fashion_data:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        
        if self.transform is not None:
            pil_image = Image.fromarray(np.uint8(image)) 
            image = self.transform(pil_image)
            
        return image, label