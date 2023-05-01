# Source :
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
# https://github.com/nicknochnack/YOLO-Drowsiness-Detection/blob/main/Drowsiness%20Detection%20Tutorial.ipynb
# https://github.com/tzutalin/labelImg


# Label Image - "myhand" environment
# -----------
# cd C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\labelImg
# python labelImg.py

# YoLoV5 Training - "mybrain" environment
# ---------------
# cd C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\yolov5
# python train.py --img 320 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --workers 2


#Install and Import Dependencies

import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import math
import time

# Create CSV file
import csv

header_ang = [ 'timer', 'timer_task', 
            'angle01', 'angle02', 'angle03', 
            'angle11', 'angle12', 'angle13',
            'angle21', 'angle22', 'angle23', 
            'angle31', 'angle32', 'angle33',
            'angle41', 'angle42', 'angle43', 'task_state', 'thumb_state', ] 
        
csvfile_ang = open(r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\hand_angle.csv', 'w')
writer_ang = csv.writer(csvfile_ang, delimiter = ',', lineterminator='\n')
writer_ang.writerow(header_ang)

header_sta = [ 'timer', 'timer_task', 'grasp_type', 'rotate_type', 'box_near_hand'] 
        
csvfile_sta = open(r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\hand_status.csv', 'w')
writer_sta = csv.writer(csvfile_sta, delimiter = ',', lineterminator='\n')
writer_sta.writerow(header_sta)




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SETUP Graph Neural Networks

import torch
import torch.nn.functional as F
from torch.nn import Linear

dataset_num_node_features = 1
dataset_num_classes = 7

# Training with GCNConv
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

""""""
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset_num_node_features, hidden_channels)        # dataset.num_node_features
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset_num_classes)                 # dataset.num_classes

    def forward(self, x, edge_index, batch): #
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


# Training with GraphConv

from torch_geometric.nn import GraphConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GraphConv(dataset_num_node_features, hidden_channels)        # dataset.num_node_features
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset_num_classes)                 # dataset.num_classes

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Defining ANN Architechture
model_gnn = GNN(hidden_channels=64)
model_gnn.load_state_dict(torch.load(r"C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\model_grasppose_gnn.pkl"))
model_gnn.to(device)
model_gnn.eval()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.01)

# Data Preprocessing
edge_index = torch.tensor([[0, 1],          #[1, 0],
                           [1, 2],          #[2, 1],
                           [2, 3],          #[3, 2],
                           [0, 4],          #[4, 0],
                           [4, 5],          #[5, 4],
                           [5, 6],          #[6, 5],
                           [0, 7],          #[7, 0],
                           [7, 8],          #[8, 7],
                           [8, 9],          #[9, 8],
                           [0, 10],         #[10, 0],
                           [10, 11],        #[11, 10],
                           [11, 12],        #[12, 11],
                           [0, 13],         #[13, 0],
                           [13, 14],        #[14, 13],
                           [14, 15]         #[15, 14] 
                           ],dtype=torch.long)





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SETUP NEURAL NETWORKS RNN

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)

        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)           # <<<<<<<<<<< RNN
        # or:
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)            # <<<<<<<<<<< GRU
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)         # <<<<<<<<<<< LSTM
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)          # <<<<<<<<<<< LSTM
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)                                                            # <<<<<<<<<<< RNN
        # or:
        #out, _ = self.lstm(x, (h0,c0))                                                      # <<<<<<<<<<< LSTM
        # or:
        out, _ = self.gru(x, h0)                                                            # <<<<<<<<<<< GRU
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_classes = 7
num_epochs = 50
batch_size = 1
learning_rate = 0.001

input_size = 15
sequence_length = 10
hidden_size = 128
num_layers = 2

# Defining ANN Architechture
model_rnn = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model_rnn.load_state_dict(torch.load(r"C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\model_gru.pkl"))
model_rnn.to(device)
model_rnn.eval()

import collections
coll_hand = collections.deque(maxlen=sequence_length)

import pickle
sc_input = pickle.load(open(r"C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\scaler_input.pkl",'rb'))





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HAND TRACKING : MEDIAPIPE

### HAND TRACKING: SETUP

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

joint_list_0 = [[2,1,0], [3,2,1], [4,3,2]]
joint_list_1 = [[6,5,0], [7,6,5], [8,7,6]]
joint_list_2 = [[10,9,0], [11,10,9], [12,11,10]]
joint_list_3 = [[14,13,0], [15,14,13], [16,15,14]]
joint_list_4 = [[18,17,0], [19,18,17], [20,19,18]]

### HAND TRACKING: FUCTION 

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
            
    return output



def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:

            point = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord

            vector_A = np.array( [ a[0]-b[0], a[1]-b[1] , a[2]-b[2] ])
            vector_B = np.array( [ c[0]-b[0], c[1]-b[1] , c[2]-b[2] ])

            length_A = math.sqrt( pow(a[0]-b[0],2) + pow(a[1]-b[1],2) + pow(a[2]-b[2],2) )
            length_B = math.sqrt( pow(c[0]-b[0],2) + pow(c[1]-b[1],2) + pow(c[2]-b[2],2) )      

            radians = math.acos( np.dot(vector_A, vector_B) / (length_A * length_B) )
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(point, [1920, 1080]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    return image


def get_finger_angles(results, joint_list):
    
    finger_angles=[]

    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        
        joint_no = 1
        for joint in joint_list:

            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord
            
            vector_A = np.array( [ a[0]-b[0], a[1]-b[1] , a[2]-b[2] ])
            vector_B = np.array( [ c[0]-b[0], c[1]-b[1] , c[2]-b[2] ])

            length_A = math.sqrt( pow(a[0]-b[0],2) + pow(a[1]-b[1],2) + pow(a[2]-b[2],2) )
            length_B = math.sqrt( pow(c[0]-b[0],2) + pow(c[1]-b[1],2) + pow(c[2]-b[2],2) )      

            radians = math.acos( np.dot(vector_A, vector_B) / (length_A * length_B) )
            angle = np.abs(radians*180.0/np.pi)
            
            #if joint_no == 1 and angle < 90 :
            #    angle = 90
            #elif joint_no == 2 and angle < 110 :
            #    angle = 110
            #elif joint_no == 3 and angle < 90 :
            #    angle = 90
            
            joint_no = joint_no + 1
            finger_angles.append(round(angle, 2))

    return finger_angles







# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Model and File

# Load Model
PATH_MODEL = r"C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\yolov5\runs\train\exp7\weights\best.pt"
model_yolo = torch.hub.load(r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\yolov5', 'custom', path=PATH_MODEL, force_reload=True, source='local')

#model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH_MODEL, force_reload=True)


# Make Detection
#PATH_VIDEO = r"C:\Users\anomt\Desktop\BDT\VIDEO_EXPERIMENT\NCGG\original\01_top.mp4" #top_view.mp4   tes03x
#PATH_VIDEO = r"C:\Users\anomt\Desktop\BDT\VIDEO_EXPERIMENT\NCGG\reduce\03_top.mp4"
#PATH_VIDEO = r"C:\Users\anomt\Desktop\BDT\VIDEO_GESTURE\TUMB\down_02.mp4"

PATH_VIDEO = r"C:\Users\anomt\Desktop\BDT\VIDEO_EXPERIMENT\TLL\TOP_VIEW\anom_01.mp4"

#PATH_VIDEO = r"C:\Users\anomt\Desktop\Block_Grasping.mp4"
#PATH_VIDEO = r"C:\Users\anomt\Desktop\Rotation\GRASP\4.mp4"

cap = cv2.VideoCapture(PATH_VIDEO) #  PATH_VIDEO 0

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (960,540) ) #(960,540) (640,480)

face_01 = cv2.imread(r"C:\Users\anomt\Desktop\BDT\FILES\LABEL_50x50\01.png", cv2.IMREAD_COLOR)
face_02 = cv2.imread(r"C:\Users\anomt\Desktop\BDT\FILES\LABEL_50x50\02.png", cv2.IMREAD_COLOR)
face_03 = cv2.imread(r"C:\Users\anomt\Desktop\BDT\FILES\LABEL_50x50\03.png", cv2.IMREAD_COLOR)
face_04 = cv2.imread(r"C:\Users\anomt\Desktop\BDT\FILES\LABEL_50x50\04.png", cv2.IMREAD_COLOR)
face_05 = cv2.imread(r"C:\Users\anomt\Desktop\BDT\FILES\LABEL_50x50\05.png", cv2.IMREAD_COLOR)
face_06 = cv2.imread(r"C:\Users\anomt\Desktop\BDT\FILES\LABEL_50x50\06.png", cv2.IMREAD_COLOR)

#size = 40
#face_01 = cv2.resize(face_01, (size, size))
#face_02 = cv2.resize(face_02, (size, size))
#face_03 = cv2.resize(face_03, (size, size))
#face_04 = cv2.resize(face_04, (size, size))
#face_05 = cv2.resize(face_05, (size, size))
#face_06 = cv2.resize(face_06, (size, size))

gray_face_01 = cv2.cvtColor(face_01, cv2.COLOR_BGR2GRAY)
gray_face_02 = cv2.cvtColor(face_02, cv2.COLOR_BGR2GRAY)
gray_face_03 = cv2.cvtColor(face_03, cv2.COLOR_BGR2GRAY)
gray_face_04 = cv2.cvtColor(face_04, cv2.COLOR_BGR2GRAY)
gray_face_05 = cv2.cvtColor(face_05, cv2.COLOR_BGR2GRAY)
gray_face_06 = cv2.cvtColor(face_06, cv2.COLOR_BGR2GRAY)

ret_01, mask_face_01 = cv2.threshold(gray_face_01, 1, 255, cv2.THRESH_BINARY)
ret_02, mask_face_02 = cv2.threshold(gray_face_02, 1, 255, cv2.THRESH_BINARY)
ret_03, mask_face_03 = cv2.threshold(gray_face_03, 1, 255, cv2.THRESH_BINARY)
ret_04, mask_face_04 = cv2.threshold(gray_face_04, 1, 255, cv2.THRESH_BINARY)
ret_05, mask_face_05 = cv2.threshold(gray_face_05, 1, 255, cv2.THRESH_BINARY)
ret_06, mask_face_06 = cv2.threshold(gray_face_06, 1, 255, cv2.THRESH_BINARY)

####################

img_00 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\00.jpg')
img_01 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\01.jpg')
img_02 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\02.jpg')
img_03 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\03.jpg')
img_04 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\04.jpg')
img_05 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\05.jpg')
img_06 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\06.jpg')
img_07 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\07.jpg')
img_08 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\08.jpg')
img_09 = cv2.imread(r'C:\Users\anomt\Desktop\BDT\FILES\TEST_100x100\09.jpg')

gray_img_00 = cv2.cvtColor(img_00, cv2.COLOR_BGR2GRAY)
gray_img_01 = cv2.cvtColor(img_01, cv2.COLOR_BGR2GRAY)
gray_img_02 = cv2.cvtColor(img_02, cv2.COLOR_BGR2GRAY)
gray_img_03 = cv2.cvtColor(img_03, cv2.COLOR_BGR2GRAY)
gray_img_04 = cv2.cvtColor(img_04, cv2.COLOR_BGR2GRAY)
gray_img_05 = cv2.cvtColor(img_05, cv2.COLOR_BGR2GRAY)
gray_img_06 = cv2.cvtColor(img_06, cv2.COLOR_BGR2GRAY)
gray_img_07 = cv2.cvtColor(img_07, cv2.COLOR_BGR2GRAY)
gray_img_08 = cv2.cvtColor(img_08, cv2.COLOR_BGR2GRAY)
gray_img_09 = cv2.cvtColor(img_09, cv2.COLOR_BGR2GRAY)

ret_img_00, mask_img_00 = cv2.threshold(gray_img_00, 1, 255, cv2.THRESH_BINARY)
ret_img_01, mask_img_01 = cv2.threshold(gray_img_01, 1, 255, cv2.THRESH_BINARY)
ret_img_02, mask_img_02 = cv2.threshold(gray_img_02, 1, 255, cv2.THRESH_BINARY)
ret_img_03, mask_img_03 = cv2.threshold(gray_img_03, 1, 255, cv2.THRESH_BINARY)
ret_img_04, mask_img_04 = cv2.threshold(gray_img_04, 1, 255, cv2.THRESH_BINARY)
ret_img_05, mask_img_05 = cv2.threshold(gray_img_05, 1, 255, cv2.THRESH_BINARY)
ret_img_06, mask_img_06 = cv2.threshold(gray_img_06, 1, 255, cv2.THRESH_BINARY)
ret_img_07, mask_img_07 = cv2.threshold(gray_img_07, 1, 255, cv2.THRESH_BINARY)
ret_img_08, mask_img_08 = cv2.threshold(gray_img_08, 1, 255, cv2.THRESH_BINARY)
ret_img_09, mask_img_09 = cv2.threshold(gray_img_09, 1, 255, cv2.THRESH_BINARY)

####################

n_frame = 0    
n_capture = 1            # Normal 3  # Realtime 7
#n_contour = 0
n_test = 0

timer_task_all = []
timer_return = 0

timer_task_01 = 0
timer_task_02 = 0
timer_task_03 = 0
timer_task_03 = 0
timer_task_04 = 0
timer_task_05 = 0
timer_task_06 = 0
timer_task_07 = 0
timer_task_08 = 0

timer_flag_01 = True
timer_flag_02 = True
timer_flag_03 = True
timer_flag_04 = True
timer_flag_05 = True
timer_flag_06 = True
timer_flag_07 = True
timer_flag_08 = True

ans_01 = [2,1,1,2]
ans_02 = [1,3,1,1]
ans_03 = [2,2,3,4]
ans_04 = [5,1,4,1]
ans_05 = [4,3,5,6]
ans_06 = [1,4,6,1]
ans_07 = [5,6,4,3]
ans_08 = [5,3,4,5]

grasp_pose = [0,0,0,0,0,0,0]

start_zero = time.time()
start = time.time()



with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        ret, frame_next = cap.read()

        if ret:
            
            f_height, f_width, f_channel = frame.shape

            #width = 960     # int(img.shape[1] * scale_percent / 100) 
            #height = 540    # int(img.shape[0] * scale_percent / 100)
            #dim = (width, height)
            #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            if(n_frame % n_capture == 0 ):

                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hand Detection

                timer_task = round (time.time() - start , 2)

                grasp_type = None
                rotate_type = None 
                box_near_hand = None

                task_state = 0
                thumb_state = 0


                # Brightness and Contrast
                #alpha = 1.5
                #beta = 5
                #frame = cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)
                
                # BGR 2 RGB
                frame_hand = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)
                # Set flag
                frame_hand.flags.writeable = False
                # Hand Detections
                results = hands.process(frame_hand)
                # Set flag to true
                frame_hand.flags.writeable = True
                # RGB 2 BGR
                frame_hand = cv2.cvtColor(frame_hand, cv2.COLOR_RGB2BGR)

                hand_status = False
                hand_angle = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                hand_position = [0,0,0,0,0,0,0,0,0]
                stream = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

                list_hand_x = []
                list_hand_y = []
                
                ### If hand detected

                # Rendering results
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, 
                                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=3, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=3, circle_radius=2),
                                                )
                            
                        # Render left or right detection
                        #if get_label(num, hand, results):
                        #    text, coord = get_label(num, hand, results)
                        #    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 1, cv2.LINE_AA)
                    
                    hand_status = True

                    ### Measure Angle 
                    # Draw angles to image from joint list
                    draw_finger_angles(frame, results, joint_list_0)
                    draw_finger_angles(frame, results, joint_list_1)
                    draw_finger_angles(frame, results, joint_list_2)
                    draw_finger_angles(frame, results, joint_list_3)
                    draw_finger_angles(frame, results, joint_list_4)

                    angle_0 = get_finger_angles(results, joint_list_0)
                    angle_1 = get_finger_angles(results, joint_list_1)
                    angle_2 = get_finger_angles(results, joint_list_2)
                    angle_3 = get_finger_angles(results, joint_list_3)
                    angle_4 = get_finger_angles(results, joint_list_4)

                    hand_angle = [  angle_0[0], angle_0[1], angle_0[2],
                                    angle_1[0], angle_1[1], angle_1[2],
                                    angle_2[0], angle_2[1], angle_2[2],   
                                    angle_3[0], angle_3[1], angle_3[2],
                                    angle_4[0], angle_4[1], angle_4[2] ]

                    timer = round (time.time() - start_zero , 2) 
                    
                    #writer_ang.writerow([   timer, timer_task, 
                    #                    angle_0[0], angle_0[1], angle_0[2],
                    #                    angle_1[0], angle_1[1], angle_1[2],
                    #                    angle_2[0], angle_2[1], angle_2[2],   
                    #                    angle_3[0], angle_3[1], angle_3[2],
                    #                    angle_4[0], angle_4[1], angle_4[2], thumb_state ])

                    #print( str(timer) + " - " + str(hand_angle) )

                    ### Measure Distance
                    
                    # Create new variabel for wrist 
                    wrist = np.array( [hand.landmark[9].x, hand.landmark[9].y] )

                    # Create new variabel for fingertip
                    tip_0 = np.array([hand.landmark[4].x, hand.landmark[4].y] ) # , hand.landmark[4].z
                    tip_1 = np.array([hand.landmark[8].x, hand.landmark[8].y] ) # , hand.landmark[8].z
                    tip_2 = np.array([hand.landmark[12].x, hand.landmark[12].y] ) # , hand.landmark[12].z
                    tip_3 = np.array([hand.landmark[16].x, hand.landmark[16].y] ) # , hand.landmark[16].z
                    tip_4 = np.array([hand.landmark[20].x, hand.landmark[20].y] ) # , hand.landmark[20].z
                           
                    # Area of Hand
                    
                    for i in range(21):
                        list_hand_x.append(hand.landmark[i].x)
                        list_hand_y.append(hand.landmark[i].y)
                    
                    min_hand_x = int (min(list_hand_x) * f_width)
                    min_hand_y = int (min(list_hand_y) * f_height)

                    max_hand_x = int (max(list_hand_x) * f_width)
                    max_hand_y = int (max(list_hand_y) * f_height)
                    
                    cv2.rectangle(frame, (min_hand_x, min_hand_y),(max_hand_x, max_hand_y), (255, 0, 0), 2)  




                    # Drawing circle in fingertip
                    """
                    frame = cv2.circle(frame, ( int (hand.landmark[4].x * vwidth), 
                                                int (hand.landmark[4].y * vheight)), 
                                                radius=10, color=(0, 0, 100), thickness=-1)
                    
                    frame = cv2.circle(frame, ( int (hand.landmark[8].x * vwidth), 
                                                int (hand.landmark[8].y * vheight)), 
                                                radius=10, color=(0, 0, 100), thickness=-1)

                    frame = cv2.circle(frame, ( int (hand.landmark[12].x * vwidth), 
                                                int (hand.landmark[12].y * vheight)), 
                                                radius=10, color=(0, 0, 100), thickness=-1)
                    
                    frame = cv2.circle(frame, ( int (hand.landmark[16].x * vwidth), 
                                                int (hand.landmark[16].y * vheight)), 
                                                radius=10, color=(0, 0, 100), thickness=-1)
                    
                    frame = cv2.circle(frame, ( int (hand.landmark[20].x * vwidth), 
                                                int (hand.landmark[20].y * vheight)), 
                                                radius=10, color=(0, 0, 100), thickness=-1)
                    """

                    
                    
                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PREDICT ACTION

                    #stream = stream.decode().split(',')
                    #stream = [eval(i) for i in stream]        #round((eval(i)/3.14), 2)

                    stream[0] = angle_0[0] / 180
                    stream[1] = angle_0[1] / 180
                    stream[2] = angle_0[2] / 180
                    stream[3] = angle_1[0] / 180
                    stream[4] = angle_1[1] / 180
                    stream[5] = angle_1[2] / 180
                    stream[6] = angle_2[0] / 180
                    stream[7] = angle_2[1] / 180
                    stream[8] = angle_2[2] / 180
                    stream[9] = angle_3[0] / 180
                    stream[10] = angle_3[1] / 180
                    stream[11] = angle_3[2] / 180
                    stream[12] = angle_4[0] / 180
                    stream[13] = angle_4[1] / 180
                    stream[14] = angle_4[2] / 180

                    x = torch.tensor([  [1],
                                        [stream[0]], [stream[1]], [stream[2]], 
                                        [stream[3]], [stream[4]], [stream[5]],
                                        [stream[6]], [stream[7]], [stream[8]],
                                        [stream[9]], [stream[10]], [stream[11]],
                                        [stream[12]], [stream[13]], [stream[14]] ], dtype=torch.float)
                    #print(x)
                    
                    data = Data(x=x, edge_index=edge_index.t().contiguous())    

                    output_gnn = model_gnn(data.x, data.edge_index, data.batch) #
                    predicted_gnn = (torch.max(torch.exp(output_gnn), 1)[1]).data.cpu().numpy()
            
                    #predicted_gnn = torch.max(output_gnn, 1)

                    probs = torch.nn.functional.softmax(output_gnn, dim=1)
                    str_probs = str( format((torch.max(probs).item()*100),".2f") )

                    if predicted_gnn.item() == 0:
                        cv2.putText(frame, "Rake " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        grasp_pose[0]+=1

                    elif predicted_gnn.item() == 1:
                        cv2.putText(frame, "Palmar Grasp " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        grasp_pose[1]+=1

                    elif predicted_gnn.item() == 2:
                        cv2.putText(frame, "Radial Palmar Grasp " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        grasp_pose[2]+=1

                    elif predicted_gnn.item() == 3:
                        cv2.putText(frame, "Radial Digital Grasp " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        grasp_pose[3]+=1

                    elif predicted_gnn.item() == 4:
                        cv2.putText(frame, "Inferior Pincher " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        grasp_pose[4]+=1
                    
                    elif predicted_gnn.item() == 5:
                        cv2.putText(frame, "Pincher " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)                 
                        grasp_pose[5]+=1

                    elif predicted_gnn.item() == 6:
                        
                        if hand.landmark[20].z > hand.landmark[4].z:
                            cv2.putText(frame, "Thumbs UP " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                            thumb_state = 1

                        else:
                            cv2.putText(frame, "Thumbs DOWN " + str_probs + "%" , (min_hand_x, max_hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2) 
                            thumb_state = -1                
                
                        grasp_pose[6]+=1

                    grasp_type = predicted_gnn.item()


                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PREDICT ACTION

                    # Feature Scaling
                    coll_hand.append(hand_angle)

                    if len(coll_hand) == sequence_length:

                        x_data = np.array(list(coll_hand))
                        x_train = sc_input.transform(x_data)
                        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
                        x_train = x_train[None, :, :]
                                
                        output_rnn = model_rnn(x_train)
                        confidence_rnn, predicted_rnn = torch.max(output_rnn.data, 1)
                        
                        str_conf = str( format(confidence_rnn.item()*10,".2f") )
                        '''
                        if predicted_rnn.item() == 0:
                            cv2.putText(frame, "No Rotation " + str_conf + "%", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted_rnn.item() == 1:
                            cv2.putText(frame, "Rotate Type 1  " + str_conf + "%", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted_rnn.item() == 2:
                            cv2.putText(frame, "Rotate Type 2  " + str_conf + "%", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted_rnn.item() == 3:
                            cv2.putText(frame, "Rotate Type 3  " + str_conf + "%", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted_rnn.item() == 4:
                            cv2.putText(frame, "Rotate Type 4 " + str_conf + "%", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted_rnn.item() == 5:
                            cv2.putText(frame, "Rotate Type 5 " + str_conf + "%", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted_rnn.item() == 6:
                            cv2.putText(frame, "Rotate Type 6 " + str_conf + "%", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2) #+ str_conf + "%"
                        '''
                        rotate_type = predicted_rnn.item()
                        #print(predicted.item())





                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Block Detection

                # Frame threshold 
                imgBlur = cv2.GaussianBlur(frame, (7,7), 1)
                imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
                ret, imgThres = cv2.threshold(imgGray, 195, 255, cv2.THRESH_BINARY)
                
                # Make detections 
                results = model_yolo(frame_next)

                df_tracked_objects = results.pandas().xyxy[0]
                list_tracked_objects = df_tracked_objects.values.tolist()
                #print(list_tracked_objects)
                    
                #if len(list_tracked_objects) == 4: #>0

                box_num = 0
                box_face = 0
                box_list = []
                box_design = []
                box_distance = []
                box_design_sort = []

                box_near_hand = []
                #avg_confidence = []

                pos_x = []
                pos_y = []

                for x1, y1, x2, y2, conf_pred, cls_id, cls in list_tracked_objects:

                    if conf_pred > 0.8:

                        #avg_confidence.append( round(conf_pred,2) )

                        center_x = int ((x1+x2)/2)
                        center_y = int ((y1+y2)/2)
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        w = int (x2-x1)
                        h = int (y2-y1)

                        box_distance.append( int (math.sqrt( pow(center_x, 2) + pow(center_y, 2) )) )
                        #print(center_x, center_y)

                        pos_x.append( int (center_x) )
                        pos_y.append( int (center_y) )
                        
                        dim = (100, 100)
                        imgBox = cv2.resize(imgThres[y1:y2, x1:x2], dim, interpolation = cv2.INTER_AREA)
                        #cv2.imshow("Box_"+str(box_num), imgBox)
                        
                        box_class = [ imgBox[50,25], imgBox[75,50], imgBox[50,75], imgBox[25,50] ]

                        if box_class == [0,0,0,0] :
                            box_face = 1
                            box_design.append(1)
                        elif box_class == [255,255,255,255]:
                            box_face = 2
                            box_design.append(2)
                        #... dipisah
                        elif box_class == [255,255,0,0]:
                            box_face = 3
                            box_design.append(3)  
                        elif box_class == [255,0,0,255]:
                            box_face = 4
                            box_design.append(4)  
                        elif box_class == [0,0,255,255]:
                            box_face = 5
                            box_design.append(5)  
                        elif box_class == [0,255,255,0]:
                            box_face = 6
                            box_design.append(6)               
                        
                        cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 255, 0), 2)

                        # Select Block Inside Hand

                        if hand_status == True:
                            if (center_x > min_hand_x-150 and center_x < max_hand_x+150 ) and (center_y > min_hand_y-150 and center_y < max_hand_y+150):
                                cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 0, 255), 5)
                                box_near_hand.append(box_face)
                                #print(box_face)
                            #else:
                            #    print(0)
                        #else:
                        #    print(0)
                            
                        cv2.putText(frame, str(round(conf_pred,2)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        #cv2.putText(frame, "id:" +str(box_num), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        box_num = box_num + 1
                        roi_label = frame[y1:y1+50, x1:x1+50]

                        if(box_face == 1):
                            roi_label [np.where(mask_face_01)] = 0
                            roi_label += face_01
                        elif(box_face == 2):
                            roi_label [np.where(mask_face_02)] = 0
                            roi_label += face_02
                        elif(box_face == 3):
                            roi_label [np.where(mask_face_03)] = 0
                            roi_label += face_03
                        elif(box_face == 4):
                            roi_label [np.where(mask_face_04)] = 0
                            roi_label += face_04
                        elif(box_face == 5):
                            roi_label [np.where(mask_face_05)] = 0
                            roi_label += face_05
                        elif(box_face == 6):
                            roi_label [np.where(mask_face_05)] = 0
                            roi_label += face_06
                        
                        # Draw objects features
                        #cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                        #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        #cv2.putText(frame, cls , (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #print(box_design)
                #print (box_near_hand)

                


                if len(box_design) == 4 and len(box_distance) == 4:

                    # >>>>>>>>>>>>

                    box_0 = (pos_x[0], pos_y[0])
                    box_1 = (pos_x[1], pos_y[1])
                    box_2 = (pos_x[2], pos_y[2])
                    box_3 = (pos_x[3], pos_y[3])

                    frame = cv2.line(frame, box_0, box_1, (0, 0, 0), 2)
                    frame = cv2.line(frame, box_0, box_2, (0, 0, 0), 2)
                    frame = cv2.line(frame, box_0, box_3, (0, 0, 0), 2)
                    frame = cv2.line(frame, box_1, box_2, (0, 0, 0), 2)
                    frame = cv2.line(frame, box_1, box_3, (0, 0, 0), 2)
                    frame = cv2.line(frame, box_2, box_3, (0, 0, 0), 2)

                    pos_x_order = [ pos_x[0], pos_x[1], pos_x[2], pos_x[3] ]
                    pos_y_order = [ pos_y[0], pos_y[1], pos_y[2], pos_y[3] ]

                    #if  ( abs(pos_x[0] - pos_x[1]) < 100) and \
                    #    ( abs(pos_x[0] - pos_x[2]) < 100) and \
                    #    ( abs(pos_x[0] - pos_x[3]) < 100) and \
                    #    ( abs(pos_x[1] - pos_x[2]) < 100) and \
                    #    ( abs(pos_x[1] - pos_x[3]) < 100) and \
                    #    ( abs(pos_x[2] - pos_x[3]) < 100):

                    #    start = time.time()
                    
                    #elif( abs(pos_y[0] - pos_y[1]) < 100) and \
                    #    ( abs(pos_y[0] - pos_y[2]) < 100) and \
                    #    ( abs(pos_y[0] - pos_y[3]) < 100) and \
                    #    ( abs(pos_y[1] - pos_y[2]) < 100) and \
                    #    ( abs(pos_y[1] - pos_y[3]) < 100) and \
                    #    ( abs(pos_y[2] - pos_y[3]) < 100):

                    #    start = time.time()

                    # >>>>>>>>>>>>

                    len_0 = int (math.sqrt( (pos_x[0]-pos_x[1])**2 + (pos_y[0]-pos_y[1])**2 ) )
                    len_1 = int (math.sqrt( (pos_x[1]-pos_x[2])**2 + (pos_y[1]-pos_y[2])**2 ) )
                    len_2 = int (math.sqrt( (pos_x[2]-pos_x[3])**2 + (pos_y[2]-pos_y[3])**2 ) )
                    len_3 = int (math.sqrt( (pos_x[3]-pos_x[0])**2 + (pos_y[3]-pos_y[0])**2 ) )
                    len_4 = int (math.sqrt( (pos_x[0]-pos_x[2])**2 + (pos_y[0]-pos_y[2])**2 ) )
                    len_5 = int (math.sqrt( (pos_x[1]-pos_x[3])**2 + (pos_y[1]-pos_y[3])**2 ) )

                    # Order Len
                    len_order = [ len_0, len_1, len_2, len_3, len_4, len_5 ]
                    len_rect = sorted(len_order)

                    if  ( abs(len_rect[0] - len_rect[1]) < 50) and \
                        ( abs(len_rect[0] - len_rect[2]) < 50) and \
                        ( abs(len_rect[0] - len_rect[3]) < 50) and \
                        ( abs(len_rect[1] - len_rect[2]) < 50) and \
                        ( abs(len_rect[1] - len_rect[3]) < 50) and \
                        ( abs(len_rect[2] - len_rect[3]) < 50):

                        # >>>>>>>>>>>>
                        
                        sort_index = sorted(range(len(box_distance)), key=lambda k: box_distance[k])

                        for i in range(len(sort_index)):
                            box_design_sort.append(box_design[sort_index[i]])
                        
                        print("----- TASK :" + str(box_design_sort))

                        if box_design_sort == ans_01:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_01)] = 0
                            test_label += img_01
                            task_state = 1
                            
                            if(timer_flag_01):
                                end = time.time()
                                timer_task_01 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_01)

                                print ("TASK 1 COMPLETED in " + str(timer_task_01) +" seconds")
                                timer_flag_01 = False
                            else:
                                print ("TASK 1 COMPLETED in " + str(timer_task_01) +" seconds")
                            start = time.time()
                            
                                                            
                        elif box_design_sort == ans_02:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_02)] = 0
                            test_label += img_02
                            task_state = 1
                            
                            if(timer_flag_02):
                                end = time.time()
                                timer_task_02 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_02)

                                print ("TASK 2 COMPLETED in " + str(timer_task_02) +" seconds")
                                timer_flag_02 = False
                            else:
                                print ("TASK 2 COMPLETED in " + str(timer_task_02) +" seconds")
                            start = time.time()
                            
                                                
                        elif box_design_sort == ans_03:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_03)] = 0
                            test_label += img_03
                            task_state = 1

                            if(timer_flag_03):
                                end = time.time()
                                timer_task_03 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_03)

                                print ("TASK 3 COMPLETED in " + str(timer_task_03) +" seconds")
                                timer_flag_03 = False
                            else:
                                print ("TASK 3 COMPLETED in " + str(timer_task_03) +" seconds")
                            start = time.time()
                            
                                            
                        elif box_design_sort == ans_04:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_04)] = 0
                            test_label += img_04
                            task_state = 1
                            
                            if(timer_flag_04):
                                end = time.time()
                                timer_task_04 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_04)
                                
                                print ("TASK 4 COMPLETED in " + str(timer_task_04) +" seconds")
                                timer_flag_04 = False
                            else:
                                print ("TASK 4 COMPLETED in " + str(timer_task_04) +" seconds")
                            start = time.time()
                            
                                                
                        elif box_design_sort == ans_05:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_05)] = 0
                            test_label += img_05
                            task_state = 1
                            
                            if(timer_flag_05):
                                end = time.time()
                                timer_task_05 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_05)

                                print ("TASK 5 COMPLETED in " + str(timer_task_05) +" seconds")
                                timer_flag_05 = False
                            else:
                                print ("TASK 5 COMPLETED in " + str(timer_task_05) +" seconds")
                            start = time.time()
                            
                                                
                        elif box_design_sort == ans_06:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_06)] = 0
                            test_label += img_06
                            task_state = 1
                            
                            if(timer_flag_06):
                                end = time.time()
                                timer_task_06 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_06)

                                print ("TASK 6 COMPLETED in " + str(timer_task_06) +" seconds")
                                timer_flag_06 = False
                            else:
                                print ("TASK 6 COMPLETED in " + str(timer_task_06) +" seconds")
                            start = time.time()
                            
                                                
                        elif box_design_sort == ans_07:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_07)] = 0
                            test_label += img_07
                            task_state = 1
                            
                            if(timer_flag_07):
                                end = time.time()
                                timer_task_07 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_07)
                                
                                print ("TASK 7 COMPLETED in " + str(timer_task_07) +" seconds")
                                timer_flag_07 = False
                            else:
                                print ("TASK 7 COMPLETED in " + str(timer_task_07) +" seconds")
                            start = time.time()
                            
                        
                        elif box_design_sort == ans_08:
                            test_label = frame[20:120, 20:120]
                            test_label [np.where(mask_img_08)] = 0
                            test_label += img_08
                            task_state = 1
                            
                            if(timer_flag_08):
                                end = time.time()
                                timer_task_08 = round((end-start - timer_return), 2)
                                timer_task_all.append(timer_task_08)
                                
                                print ("TASK 8 COMPLETED in " + str(timer_task_08) +" seconds")
                                timer_flag_08 = False
                            else:
                                print ("TASK 8 COMPLETED in " + str(timer_task_08) +" seconds")
                            start = time.time()
                            

                        else:
                            print ("NOT COMPLETE")

                        box_design = []
                        box_distance = []
                        box_design_sort = []


                        #cv2.imshow('Stream', frame)

                        #else:
                        #    cv2.imshow('Stream', frame_next)
                
                #elif len(box_design) != 0 :
                    #sort_index = sorted(range(len(box_distance)), key=lambda k: box_distance[k])

                    #for i in range(len(sort_index)):
                    #    box_design_sort.append(box_design[sort_index[i]])
                    
                #    print(box_design) #_sort
                
                #else:
                #    print("NOT DETECTED")


                    #print(str(n_frame) + " processed")
                    #cv2.imshow('Stream', np.squeeze(results.render()))


                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Put Text

                #cv2.putText(frame, "Timer ", (150, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                #cv2.putText(frame, str(timer_task) + " s", (350, 750), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

                #if(timer_flag_01 == False):
                #    cv2.putText(frame, "Task 1 : " + str(timer_task_01) + " s", (150, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #if(timer_flag_02 == False):
                #    cv2.putText(frame, "Task 2 : " + str(timer_task_02) + " s", (150, 830), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #if(timer_flag_03 == False):
                #    cv2.putText(frame, "Task 3 : " + str(timer_task_03) + " s", (150, 860), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #if(timer_flag_04 == False):
                #    cv2.putText(frame, "Task 4 : " + str(timer_task_04) + " s", (150, 890), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #if(timer_flag_05 == False):
                #    cv2.putText(frame, "Task 5 : " + str(timer_task_05) + " s", (150, 920), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #if(timer_flag_06 == False):
                #    cv2.putText(frame, "Task 6 : " + str(timer_task_06) + " s", (150, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #if(timer_flag_07 == False):
                #    cv2.putText(frame, "Task 7 : " + str(timer_task_07) + " s", (150, 980), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #if(timer_flag_08 == False):
                #    cv2.putText(frame, "Task 8 : " + str(timer_task_08) + " s", (150, 1010), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Stream', frame)

                timer = round (time.time() - start_zero , 2) 
                writer_sta.writerow([ timer, timer_task, grasp_type, rotate_type, str(box_near_hand) ])

                #if (thumb_state == 1 or task_state == 1):  # or thumb_state == -1
                #    efficacy_state = 1
                #else:
                #    efficacy_state = 0q

                if (hand_status == True):
                    writer_ang.writerow([   timer, timer_task, 
                        angle_0[0], angle_0[1], angle_0[2],
                        angle_1[0], angle_1[1], angle_1[2],
                        angle_2[0], angle_2[1], angle_2[2],   
                        angle_3[0], angle_3[1], angle_3[2],
                        angle_4[0], angle_4[1], angle_4[2], task_state, thumb_state ])
                    


                print( "GRASP :" + str(grasp_pose) + " ; TIMER :" + str(timer_task_all) )

                #n_contour = 0

            #else:
                #print(n_frame)
                #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                #cv2.imshow('Stream', frame)
                
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            n_frame = n_frame + 1

        else:
            break

cap.release()
cv2.destroyAllWindows()
