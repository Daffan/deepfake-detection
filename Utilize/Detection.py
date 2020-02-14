"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Jan 16, 2020
Record  :    Face Dection Module
"""

from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

#help(MTCNN)
#Define the model parameters here

fast_mtcnn = MTCNN(
    image_size = 100,
    margin=14,
    thresholds = [0.6, 0.7, 0.7],
    keep_all=True,
    device = 'cpu'
)

#This thresholds = [0.8, 0.9, 0.9] setting will exclude most mislabeled faces
# but fails to detect some faces when the video is dark (video 280)
#

def show_detected_frame_mtcnn(img_path, model_mtcnn = fast_mtcnn):
    """
    Detect the face from a single image and show the image with detected face.

    Params:
    model_mtcnn: face detect model in MTCNN class, see help(MTCNN)
    img_path: the path contains a image file

    """

    img = Image.open(img_path)
    boxes, probs = model_mtcnn.detect(img, landmarks = False)

    fig, ax = plt.subplots()
    print(boxes)
    for i in range(len(probs)):

        box = boxes[i]
        prob = probs[i]

        font = ImageFont.truetype('FreeMono.ttf',80)

        ImageDraw.Draw(img).rectangle(box, fill = None, outline = 'red', width = 10)
        ImageDraw.Draw(img).text((box[0],box[1]), text = '{:4f}'.format(prob), fill = 'red', width = 20, font = font)

    ax.imshow(img)

def show_detected_face_from_video_mtcnn(video_path, model_mtcnn = fast_mtcnn, stride = 50, margin = 20):
    """
    Show the faces detected from series of frames of a video by model_mtcnn.
    The number of frames decided by stride, 50 will give 6 frames. Margin extend
    the detected face by (40) pixels.

    Params:
    video_path: the path contains the video to be detected
    model_mtcnn: face detect model in MTCNN class, see help(MTCNN)
    stride: index increment between frames (50) by default
    margin: number of pixels to extend the face

    """


    v_cap = FileVideoStream(video_path).start()
    v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    fig, axes = plt.subplots(3, 2, figsize = (20,16))
    axes = axes.flatten()
    font = ImageFont.truetype('FreeMono.ttf',80)

    for i in range(v_len):

        frame = v_cap.read()

        if i%stride == 0:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            boxes, probs = model_mtcnn.detect(frame, landmarks = False)

            print(boxes)
            
            if len(probs) > 0:

                for j in range(len(probs)):

                    box = boxes[j]
                    prob = probs[j]
                    box_extended = [box[0]-margin, box[1]-margin, box[2]+margin, box[3]+margin]
                    ImageDraw.Draw(frame).rectangle(box_extended, fill = None, outline = 'red', width = 10)
                    ImageDraw.Draw(frame).text((box[0],box[1]), text = '{:4f}'.format(prob), fill = 'red', width = 20, font = font)

            axes[i//stride].imshow(frame)
            axes[i//stride].set_title(f'{video_path}_{i}')

    plt.tight_layout()
    plt.show()

    
    
def detect_face_from_image_dlib(image_path):
    
    face_detector = dlib.get_frontal_face_detector()
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    
    
def show_detected_face_from_video_dlib(video_path, stride = 50, margin = 20):
    """
    Show the faces detected from series of frames of a video by dlib face_detector.
    The number of frames decided by stride, 50 will give 6 frames. Margin extend
    the detected face by (40) pixels.

    Params:
    video_path: the path contains the video to be detected
    stride: index increment between frames (50) by default
    margin: number of pixels to extend the face

    """


    v_cap = FileVideoStream(video_path).start()
    v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    fig, axes = plt.subplots(3, 2, figsize = (20,16))
    axes = axes.flatten()
    font = ImageFont.truetype('FreeMono.ttf',80)

    face_detector = dlib.get_frontal_face_detector()

    for i in tqdm(range(v_len)):

        frame = v_cap.read()

        if i%stride == 0:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_detector(frame, 3)
            frame = Image.fromarray(frame)

            for box in boxes:

                    box_extended = [box.left()-margin, box.top()-margin, box.right()+margin, box.bottom()+margin]
                    #print(box_extended)
                    frame1 = ImageDraw.Draw(frame)
                    frame1.rectangle(box_extended, fill = None, outline = 'red', width = 10)

            axes[i//stride].imshow(frame)
            axes[i//stride].set_title(f'{video_path}_{i}')

    plt.tight_layout()
    plt.show()
