"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Jan 16, 2020
Record  :    Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import torch
import cv2
from sklearn.metrics import log_loss


from Utilize.Data import *
from Utilize.Detection import *
from Utilize.XceptionNet.xception import *
from Utilize.XceptionNet.models import *


def predict_with_model(image, model, post_function=nn.Softmax(dim=1), 
                        device = 'cuda:0', transform = 'test'):

    image = Image.fromarray(image)
    image = xception_default_data_transforms[transform](image).to(device)
    image = image.unsqueeze(0)

    output = model(image)
    output = post_function(output)

    prediction = float(output.detach().cpu().numpy()[0][1])

    return prediction, output

def predict_videos(video_df, model, model_mtcnn = fast_mtcnn, 
                    n_frames = 10, margin = 0, threshold = 0, device = 'cuda:0'):

    mean_pred_list = []
    mid_pred_list = []
    max_pred_list = []
    label_list = []

    for index in tqdm(video_df.index):

        group = video_df['group'][index]
        label = video_df['label'][index]
        video_path = os.path.join(GROUP_PATH_DIC[group], index)
        predictions, mean_pred, mid_pred, max_pred = video_file_frame_pred_mtcnn(video_path, model, model_mtcnn = model_mtcnn, 
                                                                        n_frames = n_frames, margin = margin, threshold = threshold, device = device)
        mean_pred_list.append(mean_pred)
        mid_pred_list.append(mid_pred)
        max_pred_list.append(max_pred)
        label_list.append(label)

    return mean_pred_list, mid_pred_list, max_pred_list, label_list

def visualize_pred_result(pred_list, label_list): 

    loss = log_loss([1 if x == 'FAKE' else 0 for x in label_list], pred_list) 
    print('Cross Entropy Loss = %0.4f' %(loss))

    img_test_df_sample = pd.DataFrame({'label': label_list, 'pred_label': pred_list})

    for i, d in img_test_df_sample.groupby('label'):
        d['pred_label'].plot(kind='hist',
                               figsize=(15, 5),
                               bins=40,
                               alpha=0.8,
                               title='Prediction')
        plt.legend(['FAKE','REAL'])
    plt.show()


def video_file_frame_pred_mtcnn(video_path, model, model_mtcnn = fast_mtcnn,
                          start_frame=1, end_frame=290,
                          device = 'cuda:0', n_frames=10, 
                          margin = 0, threshold = 0):
    """
    Predict and give result as numpy array
    """
    pred_frames = [int(round(x)) for x in np.linspace(start_frame, end_frame, n_frames)]
    predictions = []
    # print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    # Frame numbers and length of output video
    frame_num = 0
    boxes = None
    
    while reader.isOpened():
        
        _, image = reader.read()
        
        if image is None:
            break
            
        frame_num += 1
        
        if frame_num in pred_frames:
            cropped_face = crop_face_from_frame(image, model_mtcnn,
                                                margin = margin, threshold = threshold)
            if type(cropped_face) is not type(None):
                prediction, _ = predict_with_model(cropped_face, model, post_function=nn.Softmax(dim=1), 
                                    device = device)
                predictions.append(prediction)
            else:
                # If not detected, try the next frame
                pred_frames.append(frame_num + 1)

        if frame_num >= end_frame:
            break
            
    if len(predictions) == 0:
        return predictions, 0.5, 0.5, 0.5

    else:
        mid_pred = sorted(predictions)[len(predictions) // 2]
        mean_pred = np.mean(predictions)
        max_pred = np.max(predictions)
        return predictions, mean_pred, mid_pred, max_pred