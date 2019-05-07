import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import sys
from copy import copy


class FoodDataset(data.Dataset):
  """Food dataset CVPR challenge.   
  Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
  """

  def __init__(self, root, metadata_file, num_labels=211, transform=None, target_transform=None,
               loader=default_loader, test=False, min_img_bytes=4792):
    sys.stdout.flush()
    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    self.test = test
    self.root = root
    self.num_labels = num_labels
    self.images_ = OrderedDict()
    self.images = OrderedDict()
    self.labels_ = OrderedDict()
    self.labels = OrderedDict()
    
    self.metadata_file = metadata_file
    self.images_, self.labels_ = parse_info(self.metadata_file, root, istest=self.test, num_labels=args.num_labels)
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.corrupt = 0
    self.corrupt_ids = set()
     
    # Remove corrupt image files
    ids = self.images_.keys()
#     for i in tqdm(ids):
#         ## Correct but slow
#         try:
#             img = self.loader(self.images_[i])
#             img.close()
#         except:
#             self.corrupt += 1
#             self.corrupt_ids.add(i)
        ## Optimistic 
#         if os.path.getsize(self.images_[i]) < min_img_bytes:
#             self.corrupt += 1
#             self.corrupt_ids.add(i)
#        pass

    for i in ids:
        if i not in self.corrupt_ids:
            self.images[i] = copy(self.images_[i])
            if not self.test:
                self.labels[i] = copy(self.labels_[i])
    self.image_ids = list(self.images.keys())
    
#     if not self.test:
#         self.labelinfo = get_labelinfo(self.labels)
    
  def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if not self.test:
            path, target = self.images[self.image_ids[index]], self.labels[self.image_ids[index]]
        else:
            path = self.images[self.image_ids[index]]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if not self.test:
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        if self.test:
            return sample
        else:
            return sample, target

  def __len__(self):
    return len(self.images)
  
  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Number of corrupt datapoints discarded: {}\n'.format(self.corrupt)
#     if not self.test:
#         fmt_str += '    Number of labels: {}\n'.format(self.labelinfo.count)
    fmt_str += '    Root Location: {}\n'.format(self.root)
    fmt_str += '    Metadata file: {}\n'.format(self.metadata_file)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    if not self.test:
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Loader: '
    fmt_str += '\n{0}{1}'.format(tmp, self.loader.__name__)
    return fmt_str