"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Jan 16, 2020
Record  :
"""

#Load the packages

#System packages
import os
import subprocess
from os.path import join
from IPython.display import HTML
from base64 import b64encode
import random

#External packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

#Computer Vision packages and deeplearning
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#Costum packages
from Utilize.Detection import *

# Basics operation of videos: display frames, play videos and extract faces 

def display_image_from_video_list(video_path_list):
    '''
    input: video_path_list - path for video
    process:
        0. for each video in the video path list
        1. perform a video capture from the video
        2. read the image
        3. display the image
    '''
    plt.figure()
    fig, ax = plt.subplots(2,3,figsize=(16,8))
    # we only show images extracted from the first 6 videos
    for i, video_file in enumerate(video_path_list[0:6]):
        video_path = os.path.join(video_file)
        capture_image = cv2.VideoCapture(video_path)
        ret, frame = capture_image.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax[i//3, i%3].imshow(frame)
        ax[i//3, i%3].set_title(f"Video: {video_file}")
        ax[i//3, i%3].axis('on')

def play_video(video_path):
    '''
    Display video
    param: video_path - path of the video
    '''
    video_url = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)

# Suporting Function in cropping the videos
def add_margin(box, margin, width, height): 
    '''
    Given a box indicting the the position of cropped face, add margin to that box
    param: box a list with four numbers indicting x1, x2, y1, y2
    return: modified box
    '''
    box = [int(x) for x in box]
                
    box[0] = (np.array(box[0]-margin).clip(0, width)).tolist()
    box[2] = (np.array(box[2]+margin).clip(0, width)).tolist()
    box[1] = (np.array(box[1]-margin).clip(0, height)).tolist()
    box[3] = (np.array(box[3]+margin).clip(0, height)).tolist()

    return box


def crop_face_from_frame(image, model_mtcnn = fast_mtcnn, margin = 0, threshold = 0):

    '''
    Given a full image, use mtcnn face detection model to detect a face and return the cropped face image
    params: 
        image: [ndarray] -- input image to be cropped
        model_mtcnn: [model_mtcnn] -- default value defined in Utilize.Detection
        margin: [int] -- margin to be added to the cropped face
        threshold: [int] -- give up the cropped face if the (width + height)/2 is smaller than threshold
    '''

    cropped_face = None
    height, width = image.shape[:2]

    # detect the face
    boxes, probs = model_mtcnn.detect(Image.fromarray(image), landmarks = False)
    # if detected, add amrgin, filter by size
    if type(boxes) != type(None):

        for box in boxes:

            avg_size = (box[2]+box[3]-box[0]-box[1])/2

            if avg_size > threshold:

                box = add_margin(box, margin, width, height)
                cropped_face = image[box[1]:box[3], box[0]:box[2]]

                break

    return cropped_face

def generate_random_image(transform = 'test'):
    '''
    Generate a random image, transform if necessary
    '''
    im = Image.new('RGB', (299,299))

    def ran():
        return os.urandom(600*600)

    pixels = zip(ran(), ran(), ran())
    im.putdata(list(pixels))

    if transform is not None:
        im = xception_default_data_transforms[transform](im)

    return im 

def load_frame(frame, face_path, frame_count, missing_list = [], model_mtcnn = fast_mtcnn, missing_list_new = []):
    '''
    Load a frame 
    params:
        frame: [ndarray] -- input frame from video
        face_path: [str] -- str path refer to the folder containing all the cropped face
        frame_count: current frame number when reading the frame from video
        model_mtcnn: [model_mtcnn] -- default value defined in Utilize.Detection
        missing_list: [list] list constains the index of frames that are not able to detect a face
        missing_list_new: if new frames fail to detect a face, add it in missing_list_new
    return:
        cropped_face: [ndarray] -- cropped face
        flage: [boolen] True if fail to load a face, False if succeed
    '''
    # try read the face from this frame
    cropped_face = cv2.imread(face_path)

    # if succeed, return the face
    if cropped_face is not None: 
        return cropped_face, False

    elif ('%04d' %(frame_count)) not in missing_list:
        cropped_face = crop_face_from_frame(frame, model_mtcnn)
        if type(cropped_face) == type(None):
            missing_list_new.append('%04d' %(frame_count))
            return None, True
        else:
            cv2.imwrite(face_path, cropped_face)
            return cropped_face, False
    else:
        return None, True

def extract_face_from_video(video_path, output_path, 
                            model_mtcnn = fast_mtcnn, 
                            start_frame=1, end_frame=290, 
                            n_frames=10, margin = 0, threshold = 0):
    """
    detect video frames with mtcnn model and save the cropped face into image files
    
    Keyword Arguments
    
    """
    pred_frames = [int(round(x)) for x in np.linspace(start_frame, end_frame, n_frames)]
    
    # Read and write
    reader = cv2.VideoCapture(video_path)
    video_fn = video_path.split('/')[-1].split('.')[0]

    # Frame numbers and length of output video
    frame_num = 0
    
    while reader.isOpened():
        
        _, image = reader.read()
        
        if image is None:
            break
            
        frame_num += 1
        
        if frame_num in pred_frames:
            cropped_face = crop_face_from_frame(image, model_mtcnn = fast_mtcnn, 
                                                margin = margin, threshold = threshold)
            if type(cropped_face) is not type(None):
                status = cv2.imwrite(os.path.join(output_path, video_fn + 
                                    '_{:04d}.jpg'.format(frame_num)), cropped_face)
            else:
                # If not detected, try the next frame
                pred_frames.append(frame_num + 1)

        if frame_num >= end_frame:
            break


def video_file_multiframe(video_path, face_base_path, missing_list = [],
                                model_mtcnn = fast_mtcnn, transform = 'test', n_frames=10, 
                                margin = 0, frame_interval = 10, outlier = 15, threshold = [40, 300]):
    """
    Get the face image list from a video
    params:
        video_path: [str] -- path refers to a video
        face_base_path: [str] -- path refers to the folder contains all cropped face
        missing_list: [list] -- list contains the indexs of frames that fail to detect a face
        model_mtcnn: [model_mtcnn] -- model to detect a face
        transform: [str] -- key word of xception_default_data_transforms to decide whether to transform the face image
        n_frames: [int] -- detect 10 face in total
        frame_interval: [int] -- try to detect face every frame_interval frames, if fail, try the next frame
        outlier: [int] -- when detected required number of frames, drop the items with size outlier larger than the average size
        threshold: [list, dtype = int]: -- drop the face with size smaller than 40 and larger than 300
    return:
        missing_list_new: [list] -- contains the indexs of all the frames failed to detect a face and not in the missing_list
        size_list: [list] list of size of all the images
        image_list: [list] list contains all the image. if transform is None, the item will show in PIL format; otherwise, in ndarray
        frame_count: [list] index of where the frame read stopped

    """
    video_fn = video_path.split("/")[-1].split('.')[0]
    predictions = []
    # print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    # Frame numbers and length of output video
    boxes = None
    
    image_list = []
    size_list = []
    missing_list_new = []
    frame_count = 0
    flag = False

    # start to read the frame
    while reader.isOpened():
        _, frame = reader.read()
        
        # break if video ends
        if frame is None:
            break
        # start to crop the face if it is the frame or failed to crop a face in the previous frame
        if frame_count%frame_interval == 0 or flag: 

            flag = False
            cropped_face = None
            face_path = os.path.join(face_base_path, video_fn+'_%04d' %(frame_count)+'.jpg')

            # Load this frame, if failed, cropped face is None, flag if True
            cropped_face, flag = load_frame(frame, face_path, frame_count, missing_list, model_mtcnn = model_mtcnn, missing_list_new = missing_list_new)

            # If successfully load 
            if not flag:
                
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                width, height = cropped_face.shape[:2]
                cropped_face = Image.fromarray(cropped_face)

                # transform the image
                if transform is not None:
                    cropped_face = xception_default_data_transforms[transform](cropped_face)
                # filter by size decided by threshold
                if (width+height)/2. >threshold[0] or (width+height)/2. <threshold[1]:
                    image_list.append(cropped_face)
                    size_list.append((width+height)/2.) 
                else:
                    flag = True
        # if detected the required number, drop outliers 
        if len(size_list) == n_frames:

            selected_index = (np.abs(np.array(size_list)-np.array(size_list).mean())<outlier).tolist()
            image_list = [image_list[k] for k, boolen in enumerate(selected_index) if boolen]
            size_list = [size_list[k] for k, boolen in enumerate(selected_index) if boolen] 
        # if there is nothing to drop, finish the detecting
        if len(size_list) == n_frames:
            break
        frame_count = frame_count +1
    
    return missing_list_new, size_list, image_list, frame_count

# Default transforms
xception_default_data_transforms = {
    'ResNet': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'Resnest': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class dataset_face(Dataset):
    """Deepfake dataset"""
    
    def __init__(self, video_df, dataset_folder, split = 'train', transform = 'test', device = 'cpu'):
        """
        Initialize the dataset
        
        Keyword Arguments:
        dataset_folder: {str} -- the folder contains all the cropped face data and videos
        transfrom: {boolean} -- to transfrom the image? 
        """
        self.dataset_folder = dataset_folder
        self.video_df = video_df
        self.device = device
        self.transform = transform
        
        ### Initilize image frame ###
        
        img_name = []
        img_label = []
        
        for file_name in os.listdir(dataset_folder):
        
            if file_name.split('.')[-1] == 'jpg':
                index = file_name.split('_')[0]+'.mp4'
                
                if video_df.loc[index, 'split'] == split:

                    img_name.append(file_name)
    
                    if video_df.loc[index, 'label'] == 'FAKE':
                        img_label.append(1.0)
                    else:
                        img_label.append(0.0)
                    
        self.image_df = pd.DataFrame({'imagename': img_name, 'label': img_label})
        
        
    def __len__(self):
        
        return len(self.image_df)
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.dataset_folder, self.image_df.loc[idx, 'imagename'])
        image = Image.open(img_name)
        label = self.image_df.loc[idx, 'label']
        
        if self.transform:
            
            image = xception_default_data_transforms[self.transform](image)
            
            if self.device:
                image = image.to(device)
        
        sample = {'image': image, 'label': label} 
        
        return sample
    
    def equal_weight(self, mode = 'sampleFAKE'):
    
        if mode == 'sampleFAKE':
        
            img_df = pd.concat([self.image_df.loc[self.image_df['label']==1].sample(len(self.image_df.loc[self.image_df['label']==0])),
                                self.image_df.loc[self.image_df['label']==0]])
    
        elif mode == 'expandREAL':
        
            img_df = self.image_df.copy()
            img_df = img_df.loc[img_df.loc[img_df['label']==0].index.repeat(7)].append(img_df).reset_index(drop=True)
    
        self.image_df = img_df
    
    def reload(self):
        
        img_name = []
        img_label = []
        
        for file_name in os.listdir(self.dataset_folder):
        
            if file_name.split('.')[-1] == 'jpg':
                index = file_name.split('_')[0]+'.mp4'
    
                img_name.append(file_name)
    
                if self.video_df.loc[index, 'label'] == 'FAKE':
                    img_label.append(1)
                else:
                    img_label.append(0)
                    
        self.image_df = pd.DataFrame({'imagename': img_name, 'label': img_label})

    def filter_by_size(self, threshold = 60):

        l = len(self.image_df)

        for i in range(l):

            width, height = self[i]['image'].size
            avg_size = (width+height)/2

            if avg_size < threshold:

                self.image_df = self.image_df.drop(i)

        self.image_df = self.image_df.reset_index(drop = True)

### Video Dataset Manager ###

FACE_PATH = '../../dev/shm/deepfake_all_data_set/all_data_set/face'

BASE_PATH_1 = '../../dev/shm/deepfake_all_data_set/all_data_set/'
BASE_PATH_2 = '../../dev/deepfake_all_data_set/'

GROUP_PATH_DIC = {50:'input/train_sample_videos'}

for i in range(0, 50):
    if i<24:
        GROUP_PATH_DIC[i] = BASE_PATH_1 + 'dfdc_train_part_{00}'.format(i)
    else:
        GROUP_PATH_DIC[i] = BASE_PATH_2 + 'dfdc_train_part_{00}'.format(i)
      

class dataset_video(Dataset):
    """
    Class used to manage the dataset with all videos
    """
    
    def __init__(self, json_path, p = 0.8, face_folder = FACE_PATH, initialize = None, model_mtcnn = fast_mtcnn, transform = None, n_frames = 8, frame_interval = 2):
        
        # initialize all the parameters
        self.json_path = json_path
        self.face_folder = face_folder
        self.model_mtcnn = fast_mtcnn
        self.frame_interval = frame_interval
        self.transform = transform
        self.n_frames = 8

        # if initialize, delect the json file before
        if initialize and os.path.exists(json_path):
            os.remove(json_path)
        # if json file exist, just read the json file
        if os.path.exists(json_path):

            self.video_df = pd.read_json(json_path)
            self.p = self.video_df.groupby(['split']).count()['label'][1]/self.video_df.groupby(['split']).count()['label'].sum()
        # if fail to find the json file, collect json file again
        else:
            
            video_df_all = []
            
            for i in range(0,50):
                json_path_i = os.path.join(GROUP_PATH_DIC[i], 'metadata.json')
                if os.path.exists(json_path_i):
                    video_df = pd.read_json(json_path_i).T
                    video_df['group'] = i
                    video_df_all.append(video_df)

            self.video_df = pd.concat(video_df_all)
            self.video_df['split'] = np.random.choice(['test', 'train'], 
                                                        size=len(self.video_df), p=[1-p, p])
            self.video_df['missing_list'] = None
            # check existance, make sure all the videos showing in meta exist in the folder

            for index in tqdm(self.video_df.index):
        
                group = self.video_df.loc[index, 'group']
                video_path = os.path.join(GROUP_PATH_DIC[group], index)
                self.video_df.loc[index, 'missing_list'] = [['-1']]

                if not os.path.exists(video_path):

                    self.video_df = self.video_df.drop(index)

            self.video_df.to_json(json_path)


    def __len__(self):

        return len(self.video_df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_fn = self.video_df.index[idx]
        group = self.video_df.loc[video_fn, 'group']
        label = self.video_df.loc[video_fn, 'label']
        missing_list = self.video_df.loc[video_fn, 'missing_list']

        video_path = os.path.join(GROUP_PATH_DIC[group], video_fn)

        missing_list_new, size_list, image_list, frame_count = video_file_multiframe(video_path, self.face_folder, missing_list = missing_list,
                                                                    model_mtcnn = self.model_mtcnn, n_frames = self.n_frames, 
                                                                    transform = self.transform, frame_interval = self.frame_interval
                                                                    )
        for missing in missing_list_new:
            self.video_df.loc[video_fn, 'missing_list'].append(missing)

        if len(size_list) == 0:
            image_list = [generate_random_image(transform = self.transform)]
            
        image_list = ((int(self.n_frames/len(image_list))+1)*image_list)[:self.n_frames]

        return {'image_list': image_list, 'label': label}

    def get_face_frames(self):

        self.video_df['face_frame'] = None
        
        for face in os.listdir(self.face_folder):
            
            if face.split('.')[-1] == 'jpg':
                video_name = face.split('_')[0] + '.mp4'
                idx_frame = face.split('_')[-1].split('.')[0]
                
                if self.video_df.loc[video_name, 'face_frame']:
                    self.video_df.loc[video_name, 'face_frame'].append(idx_frame)
                else:
                    self.video_df.loc[video_name, 'face_frame'] = [[idx_frame]]
        
    def extract_face_weighted(self, n_frames = 17, model_mtcnn = fast_mtcnn, 
                    start_index = 0, end_index = -1, threshold = 0):


        for index in tqdm(self.video_df.index[start_index:end_index]):

            group_num = self.video_df.loc[index, 'group']
            video_path = GROUP_PATH_DIC[group_num]+'/'+index 

            if self.video_df.loc[index, 'label'] == 'REAL':
                n_frames_weighted = int(5.4*n_frames)
            else:
                n_frames_weighted = n_frames

            extract_face_from_video(video_path, output_path = self.face_folder, threshold = threshold,
                                     n_frames = n_frames_weighted, model_mtcnn = model_mtcnn)

    def extract_face(self, n_frames = 17, model_mtcnn = fast_mtcnn, 
                    start_index = 0, end_index = -1, threshold = 0):


        for index in tqdm(self.video_df.index[start_index:end_index]):

            group_num = self.video_df.loc[index, 'group']
            video_path = GROUP_PATH_DIC[group_num]+'/'+index 

            extract_face_from_video(video_path, output_path = self.face_folder, threshold = threshold,
                                     n_frames = n_frames, model_mtcnn = model_mtcnn, start_frame = 1)

        self.get_face_frames()
        print('Finish extracting faces!')

    def generate_face_dataset(self, split = 'train', transform = 'test', device = 'cuda:0'):

        dataset = dataset_face(self.video_df, self.face_folder, split = split,
                                transform = transform, device = device)

        return dataset

    def assign_set_by_group(self, p = 0.1,seed = 100):
    
        self.video_df.loc[self.video_df['group']==50, 'split'] = 'None'
        group_list = ['train']*(50-int(50*p))+['test']*int(50*p)
        random.Random(seed).shuffle(group_list)
        group_list = group_list+['None']
        self.video_df['split'] = [group_list[self.video_df.loc[idx, 'group']] for idx in self.video_df.index]
        self.group_list = group_list

    def select_set_by_label_same_original(self, random_state = 101):

        df = self.video_df
        df_real = df.loc[df['label'] == 'REAL']
        df_fake = pd.DataFrame({})
        for idx in df_real.index:
            t = df.loc[df['original']==idx].sample(1, random_state = random_state)
            df_fake = df_fake.append(t)
        df = pd.concat([df_real, df_fake])

        self.video_df = df

    def select_set_by_label_different_original(self, random_state = 101):

        df = self.video_df
        df_real = df.loc[df['label'] == 'REAL']
        df_real_1 = df_real.sample(frac = 0.5, random_state = 101)
        df_real_2 = df_real.drop(df_real_1.index)
        df_fake = pd.DataFrame({})
        for idx in df_real_2.index:
            t = df.loc[df['original']==idx].sample(1, random_state = random_state)
            df_fake = df_fake.append(t)
        df = pd.concat([df_real_1, df_fake])

        self.video_df = df