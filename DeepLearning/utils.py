import numpy as np
import PIL.Image

import torch
import torch.utils.data
import torchvision.transforms as transforms
import cv2

import matplotlib.pyplot as plt
import copy

TRAIN_TRANSFORMS = transforms.Compose([
    # transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CenterDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_dir, image_dir, random_hflip=True, random_vflip = True, transform=TRAIN_TRANSFORMS):
        super(CenterDataset, self).__init__()
        self.annotation_dir = annotation_dir
        self.random_hflip = random_hflip
        self.transform = transform
        self.image_dir = image_dir
                
        with open(self.annotation_dir, 'r') as f:
            self.data = [line.split() for line in f.readlines()]
            # self.data = [(image_filename: (xpos, ypos) for image_filename, xpos, ypos in f.readlines()}
                        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        jpgname, out_of_range, intersection, xpos, ypos = self.data[idx]
        out_of_range = int(1-int(out_of_range))
        intersection = int(intersection)
        xpos = int(round(float(xpos)))
        ypos = int(round(float(ypos)))

        filename = str(self.image_dir+'/'+jpgname)
        image = PIL.Image.open(filename)
        width = image.width
        height = image.height

        if self.transform is not None:
            image = self.transform(image)
        
        x = 2.0 * (xpos / width - 0.5) # map to [-1, +1]
        y = 2.0 * (ypos / height - 0.5) # map to [-1, +1]
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.flip(image, [-1])
            x = -x
            
        return image, torch.Tensor([out_of_range, intersection, x, y])


def preprocess(image: PIL.Image):
    device = torch.device('cpu')    
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]



def pointViewer(annotation_dir, image_dir, model = None, image_index = 1):

    image_to_find = '{:05d}.jpg'.format(image_index)
    image_path = str(image_dir+'/{:05d}.jpg'.format(image_index))
    print("Image : ",'{:05d}.jpg'.format(image_index))

    with open(annotation_dir, 'r') as f:
        data = [line.split() for line in f.readlines()]
    for entry in data:
        if entry[0] == image_to_find:
            label = entry[1:]
            break

    xlbl, ylbl = label
    xlbl = int(float(xlbl))
    ylbl = int(float(ylbl))

    image_ori = PIL.Image.open(image_path)
    width = image_ori.width
    height = image_ori.height
    image_np = copy.deepcopy(np.asarray(image_ori))

    cv2.circle(image_np, (xlbl, ylbl), radius=5, color=(0, 255, 0)) 

    if model is not None:
        image = TRAIN_TRANSFORMS(image_ori)
        output = model(image.unsqueeze(0)).detach()

        if output.shape != torch.Size([2]):
            output = output[0]

        xpre, ypre = output.cpu()
        xpre = ( xpre.item() / 2 + 0.5 ) * width
        ypre = ( ypre.item() / 2 + 0.5 ) * height
        cv2.circle(image_np, (int(xpre), int(ypre)), radius=5, color=(255, 0, 0))
    image = PIL.Image.fromarray(image_np)
    plt.imshow(image)
    

