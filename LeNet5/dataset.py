import os
import cv2
import numpy as np
import torch
import imutils
#from torch.utils.data import DataLoader
import config as cfg
import random
size = cfg.size

class CustomImageDataset:
    def __init__(self, txt_file):
        self.image_path = []
        self.label = []
        f = open(txt_file, "r")
        lines = f.readlines()
        for line in lines:    
            a = line.split(' ') # assign to the variable a
            self.image_path.append(a[0]) 
            a[1] = a[1].strip('\n') # remove '\n'
            self.label.append(a[1])
        
    def __len__(self):
        return len(self.label)  

    def __getitem__(self, index):
        filename = self.image_path[index]
        label = self.label[index]
        image = self.read_img(filename)
        return image, label
        

    def read_img(self, img):        
        n = cv2.imread(img)
        n = cv2.resize(n, dsize=size)#.flatten()
        return n


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        
    def __iter__(self):
        batch = []
        for idx in self.indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch





if __name__ == "__main__":
    traindataset = CustomImageDataset("train.txt") # just an object, and you need to call him
    trainloader = DataLoader(traindataset, batch_size=100, shuffle=True)

    for batch in trainloader:
        print(len(batch))
        inputs, labels = zip(*batch)

        # process the batch    
        images = inputs
        images = np.array(images) # (100, 512)
        print(images.shape) # (100, 200, 200, 3)

    


        break
        
    #print(traindataset.__len__())
    
    
