from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import customTransform
import json

class CustomDataset(Dataset):

    def __init__(self, root_dir, split, Rescale, RandomCrop, Mirror):

        self.root_dir = root_dir
        self.split = split
        self.Rescale = Rescale
        self.RandomCrop = RandomCrop
        self.Mirror = Mirror
        self.num_objects = 4
        self.objectFeaturesLen = 2048 * self.num_objects

        # Count number of elements
        num_elements = sum(1 for line in open(root_dir + 'tweet_embeddings/' + split))
        print("Number of elements in " + split + ": " + str(num_elements))

        # Initialize containers
        self.ids = np.empty(num_elements, dtype="S50")
        self.labels = np.empty(num_elements, dtype=np.float32)
        self.objectFeatures = np.zeros((num_elements, self.objectFeaturesLen), dtype=np.float32)

        # Read indices and labels

        # Read objectFeatures
        object_features_dir = "../../../datasets/EuropeanaDates/fasterRCNN/object_features/"
        for i, id in enumerate(self.ids):
            all_obj_features = np.empty(0)
            object_features_data = json.load(open(object_features_dir + id + '.json'))
            for obj_idx in range(self.num_objects):
                obj_feat = np.array(object_features_data[obj_idx]['features'])
                all_obj_features = np.concatenate((all_obj_features, obj_feat))
        self.objectFeatures[i,:] = all_obj_features


        print("Data read.")


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):

        img_name = '{}{}/{}{}'.format(self.root_dir, 'img_resized', self.ids[idx],'.jpg')
        image = Image.open(img_name)

        # Data augmentation
        width, height = image.size
        if self.RandomCrop >= width or self.RandomCrop >= height:
            image = image.resize((int(width*1.5), int(height*1.5)), Image.ANTIALIAS)

        if self.Rescale != 0:
            image = customTransform.Rescale(image,self.Rescale)

        if self.RandomCrop != 0:
            image = customTransform.RandomCrop(image,self.RandomCrop)

        if self.Mirror:
            image = customTransform.Mirror(image)

        im_np = np.array(image, dtype=np.float32)
        im_np = customTransform.PreprocessImage(im_np)


        out_img = torch.from_numpy(im_np)
        label = torch.from_numpy(self.labels[idx])
        objectFeatures = torch.from_numpy(self.objectFeatures[idx])


        return out_img, objectFeatures, label, self.ids[idx]