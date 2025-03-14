import os
import torch
import pandas as pd
import numpy as np
import PIL

from torch.utils.data import Dataset


def remove_element(x,unlabel_list):
    new_list = []
    for i in x:
        if i in unlabel_list:
            continue
        else:
            new_list.append(i)
    return new_list


class GerDataset(Dataset):

    def __init__(self,root,transform):

        unlabelled_images = [ '07081.png',
                          '02528.png',
                          '04966.png',
                          '06229.png',
                          '04285.png',
                          '03798.png',
                          '02448.png']

        unlabelled_targets = ['07081.txt',
                          '02528.txt',
                          '04966.txt',
                          '06229.txt',
                          '04285.txt',
                          '03798.txt',
                          '02448.txt']

        self.transform = transform
        self.root = root
        tmp_imgs = list(sorted(os.listdir(os.path.join(root,'images'))))
        tmp_targets = list(sorted(os.listdir(os.path.join(root,'class'))))
        self.imgs = remove_element(tmp_imgs,unlabelled_images)
        self.targets = remove_element(tmp_targets,unlabelled_targets)
    
    def __getitem__(self,idx):
        # Load images and targets
        selected_filename = self.imgs[idx]
        #print(selected_filename)
        #print(type(idx))
        imagepil = PIL.Image.open(os.path.join(self.root,'images',selected_filename)).convert('RGB')
        target = pd.read_csv(os.path.join(self.root,'class',self.targets[idx]),header=None)
        boxes = []
        #print('idx:',idx)
        #print('txt:',self.targets[idx])
        for i in range(len(target)):
            values = target[0].values[i].split()
            boxes.append([float(values[1]),float(values[2]),float(values[3]),float(values[4])])
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        # Only Ger as a label
        labels = torch.ones((len(target)),dtype=torch.int64)
        # image_id
        image_id = torch.tensor([idx])
        # Bounding Box Area
        try:
            area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        except IndexError:
            #print('boxes:',boxes)
            
            area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        # Suppose al instances are not crowd
        iscrowd = torch.zeros(len(target),dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes#.cuda()
        target['labels'] = labels#.cuda()
        target['image_id'] = image_id#.cuda()
        target['area'] = area#.cuda()
        target['iscrowd'] = iscrowd#.cuda()

        image = self.transform(imagepil)
        #image = image.cuda()
        return image, target
    
    def __len__(self):
        return len(self.imgs)


    def collate_fn(self,batch): 
        images = []
        target = []
        
        for b in batch:
            images.append(b[0])
            target.append(b[1])
            

        images = torch.stack(images, dim=0)

        return images,target  # tensor (N, 3, 300, 300), 3 lists of N tensors each
        
        
def collate_fn(batch):
    return tuple(zip(*batch))