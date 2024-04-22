import torch
import torch.utils.data as data_utl
import numpy as np
from random import randint
import os
import os.path
import glob
import cv2
import csv
import pickle


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    pic = np.asarray(pic, dtype=np.float32)
    pic = (pic/127.5) - 1

    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(frames):

    images = []
  
    for i in frames:
        img=cv2.imread(i)[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 224 or h < 224:
            d = 224.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        images.append(img)

    # images = np.asarray(images, dtype=np.float32)
    # images = (images/127.5) - 1

    return images


def load_data(normal_data_path,label_path):
    # data: N C V T M


    with open(label_path, 'rb') as f:
        sample = pickle.load(f)

    # load data
    skl = np.load(normal_data_path)

    return sample,skl



class Dataset(data_utl.Dataset):

    def __init__(self, visual_feat_dir:str,pose_feat:str,label_path:str,root:str):
        # self.data = self.make_dataset()
        self.data = []
        self.sample_duration = 64
        self.step = 2
        self.root = root
        self.vfeat_path = visual_feat_dir
        self.normal_data_path = pose_feat
        self.label_path = label_path

        # read a csv file
        dataset = []
        labels = [] # [[sub directory file path, index]]
        categories = {} # {index: category}

        # print(categories)
        sample, skeletons = load_data(self.normal_data_path,self.label_path)
        print("样本总数量：",len(sample[0]))

        for i in range(len(sample[0])):

            name = sample[0][i][0:-12]
            label = sample[1][i]
            num_frames = len(os.listdir(os.path.join(self.root, name)))
            skeleton = skeletons[i]

            dataset.append((name, label, num_frames,skeleton))

        # for vid in labels:
        #     num_frames = len(os.listdir(os.path.join('../frames', vid[0])))
                
        #     label = vid[1]
        #     dataset.append((vid[0], label, num_frames))

        self.data = dataset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf, skl = self.data[index] #骨骼数据中4000帧只不过是不断重复
        frame_indices = []
        
        images = sorted(glob.glob(self.root + vid + "/*"))
        # images = sorted(glob.glob(self.root+ vid + "/*"), key=lambda name: int(name[len(vid)+11:-5])) #按顺序读取
        
        n_frames=len(images)

        if n_frames > self.sample_duration * self.step:
            start = 0
            for i in range(start, start + self.sample_duration*self.step, self.step):
                # frame_indices.append(images[i])
                frame_indices.append(i)
        elif n_frames < self.sample_duration:
            # while len(frame_indices) < self.sample_duration:
                # frame_indices.extend(images)
            # frame_indices = frame_indices[:self.sample_duration]
            for i in range(n_frames):
                frame_indices.append(i)
            for i in range(self.sample_duration - n_frames):
                frame_indices.append(i)
        else:
            start = 0
            for i in range(start, start+self.sample_duration):
                # frame_indices.append(images[i])
                frame_indices.append(i)

        # imgs = load_rgb_frames(frame_indices)
        v_feats = np.load(os.path.join(self.vfeat_path,vid+'.npy'))


        return torch.from_numpy(np.array(frame_indices)), torch.from_numpy(v_feats), torch.from_numpy(skl), label

    def __len__(self):
        return len(self.data)