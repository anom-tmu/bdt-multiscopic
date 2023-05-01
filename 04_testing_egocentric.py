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
# python train.py --img 320 --batch 16 --epochs 100 --data dataset_ego.yaml --weights yolov5s.pt


#Install and Import Dependencies
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
import time
import datetime

# Create CSV file
import csv

header_att = [ 'timer', 'red_x', 'red_y', 'att_task', 'att_block', 'att_grasp']
                        # 0 = no att   # 1 = ok            
                        
        
csvfile_att = open(r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\hand_eye.csv', 'w')
writer_att = csv.writer(csvfile_att, delimiter = ',', lineterminator='\n')
writer_att.writerow(header_att)



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SETUP NEURAL NETWORKS RNN

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)

        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)            # <<<<<<<<<<< RNN
        # or:
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)           # <<<<<<<<<<< GRU
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
num_classes = 4
num_epochs = 100
batch_size = 1
learning_rate = 0.001

input_size = 9
sequence_data = 10
hidden_size = 128
num_layers = 2

# Defining ANN Architechture
model_nn = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model_nn.load_state_dict(torch.load(r"C:\ZOO_DATA\MYCODE\HAND\model_gru.pkl"))
model_nn.to(device)
model_nn.eval()

import collections
coll_hand = collections.deque(maxlen=sequence_data)

import pickle
sc_input = pickle.load(open(r"C:\ZOO_DATA\MYCODE\HAND\scaler_input.pkl",'rb'))





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
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [1920, 1080]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    return image


def get_finger_angles(results, joint_list):
    
    finger_angles=[]

    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        joint_no = 1
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
            
            if joint_no == 1 and angle < 90 :
                angle = 90
            elif joint_no == 2 and angle < 110 :
                angle = 110
            elif joint_no == 3 and angle < 90 :
                angle = 90
            
            joint_no = joint_no + 1
            finger_angles.append(round(angle, 2))

    return finger_angles



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

PATH_MODEL = r"C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\yolov5\runs\train\exp2\weights\best.pt"
model = torch.hub.load(r'C:\ZOO_DATA\MYCODE\HAND\YOLOv5-livinglab\yolov5', 'custom', path=PATH_MODEL, force_reload=True, source='local')

#model = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH_MODEL, force_reload=True)

#PATH_VIDEO = r"C:\Users\anomt\Desktop\BDT\VIDEO_EXPERIMENT\TLL\01_ego.mp4"

PATH_VIDEO = r"C:\Users\anomt\Desktop\BDT\VIDEO_EXPERIMENT\TLL\EGO_VIEW\chou_02.mp4"

cap = cv2.VideoCapture(PATH_VIDEO)

ret,frame=cap.read()

vheight = frame.shape[0]
vwidth = frame.shape[1]

print ("Video size", vwidth,vheight)

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (960,540)) # (960,540), (640,480)

n_frame = 0    
n_capture = 5            # Normal 3  # Realtime 7

d_wrist_comp = 0
d_timer_comp = 0
d_wrist = 0

d_attention = 0
d_attention_comp = 0

d_attention_list = [0]
d_diagonal_list = [0]

object_temp = ""

acc_block = 0
acc_phone = 0
acc_frame = 0

start_time = time.time()

hand_status = False

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        #frames +=1



        #width = 640     # int(img.shape[1] * scale_percent / 100)
        #height = 480    # int(img.shape[0] * scale_percent / 100)
        #dim = (width, height)
        #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        frame_original = frame.copy()
        
        if ret:

            if(n_frame % n_capture == 0 ):

                # Initialization variable
                att_task  = None
                att_block = None
                att_grasp = None

                red_x = 0
                red_y = 0

                acc_frame += 1 

                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visual Attention Detector
               
                # Convert to grayscale.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Blur using 3 * 3 kernel.
                gray_blurred = cv2.blur(gray, (3, 3))

                # Apply Hough transform on the blurred image.
                detected_circles = cv2.HoughCircles(gray_blurred,
                                cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                            param2 = 30, minRadius = 25, maxRadius = 27)

                # Draw circles that are detected.
                if detected_circles is not None:

                    # Convert the circle parameters a, b and r to integers.
                    detected_circles = np.uint16(np.around(detected_circles))

                    for pt in detected_circles[0, :]:
                        red_x, red_y, r = pt[0], pt[1], pt[2]

                        # Draw the circumference of the circle.
                        if red_x !=0 or red_y !=0 :
                            cv2.circle(frame, (red_x, red_y), r, (0, 0, 255), 5)
                


                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hand Detection

                # brightness and contrast
                alpha = 1.5
                beta = 5
                frame = cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)
                
                # BGR 2 RGB
                frame_hand = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
                # Set flag
                frame_hand.flags.writeable = False
                # Hand Detections
                results = hands.process(frame_hand)
                # Set flag to true
                frame_hand.flags.writeable = True
                # RGB 2 BGR
                frame_hand = cv2.cvtColor(frame_hand, cv2.COLOR_RGB2BGR)

                hand_angle = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                hand_position = [0,0,0,0,0,0,0,0,0]
                
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
                    #draw_finger_angles(frame, results, joint_list_0)
                    #draw_finger_angles(frame, results, joint_list_1)
                    #draw_finger_angles(frame, results, joint_list_2)
                    #draw_finger_angles(frame, results, joint_list_3)
                    #draw_finger_angles(frame, results, joint_list_4)

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

                    ### Measure Distance
                    
                    # Create new variabel for wrist 
                    wrist = np.array( [hand.landmark[9].x, hand.landmark[9].y] )

                    # Create new variabel for fingertip
                    tip_0 = np.array([hand.landmark[4].x, hand.landmark[4].y] ) # , hand.landmark[4].z
                    tip_1 = np.array([hand.landmark[8].x, hand.landmark[8].y] ) # , hand.landmark[8].z
                    tip_2 = np.array([hand.landmark[12].x, hand.landmark[12].y] ) # , hand.landmark[12].z
                    tip_3 = np.array([hand.landmark[16].x, hand.landmark[16].y] ) # , hand.landmark[16].z
                    tip_4 = np.array([hand.landmark[20].x, hand.landmark[20].y] ) # , hand.landmark[20].z
                    
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


                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Object Detector

                # Make detections 
                results_detection = model(frame)
                df_tracked_objects = results_detection.pandas().xyxy[0]
                list_tracked_objects = df_tracked_objects.values.tolist()
                
                name_object_list = []
                d_attention_list = []
                d_diagonal_list = []

                for x1, y1, x2, y2, conf_pred, cls_id, cls in list_tracked_objects:
                
                    if conf_pred > 0.1: #0.5
                        
                        name_object_list.append(cls)

                        center_x = int ((x1+x2)/2)
                        center_y = int ((y1+y2)/2)
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        w = int (x2-x1)
                        h = int (y2-y1)

                        if cls == "block":
                            cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 255, 0), 3)

                        elif cls == "phone":
                            cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (255, 0, 0), 3)

                        #elif cls == "hand":
                        #    cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (255, 0, 0), 3)

                        # Draw line objects to attention
                        if red_x !=0 or red_y !=0 :
                            cv2.line(frame, (center_x, center_y), (red_x, red_y), color=(0, 0, 255), thickness=2)

                        #if hand_status:
                        #   Draw line objects to hand
                        #   cv2.line(frame, (center_x, center_y), (int(wrist[0]*vwidth), int(wrist[1]*vheight)), color=(255, 0, 0), thickness=2)


                        o_length = int(x2) - int(x1)
                        o_width = int(y2) - int(y1)

                        o_diagonal = int(math.sqrt(pow(o_length,2) + pow(o_width,2)))
                        d_diagonal_list.append(o_diagonal)


                        # Detect Hand and Object -> measure speed

                        if hand_status:
                            
                            # Draw line objects to hand
                            # cv2.line(frame, (center_x, center_y), (int(wrist[0]*vwidth), int(wrist[1]*vheight)), color=(255, 0, 0), thickness=2)
                            
                            d_wrist = int (math.sqrt(pow(center_x-int(wrist[0]*vwidth),2) + pow(center_y-int(wrist[1]*vheight),2)))

                            if d_wrist_comp == 0:
                                d_wrist_comp = d_wrist
                                d_attention_comp = d_attention
                                
                                timer = round(time.time()-start_time,2)/2
                                d_timer_comp = timer
                                #print("d_timer_comp: " + str(d_timer_comp))
                            
                            else:
                                #>>>>>>>>>>>>>>>>>>>> bug
                                timer = round(time.time()-start_time,2)/2

                                if (d_timer_comp - timer) != 0:
                                    speed_hand = round(abs(d_wrist_comp - d_wrist) / abs(d_timer_comp - timer))
                                    speed_eye = round(abs(d_attention_comp - d_attention) / abs(d_timer_comp - timer))

                                    #print("Selisih timer:" + str(d_timer_comp - timer))
                                    #print(speed_hand, speed_eye)
                                    d_wrist_comp = d_wrist
                                    d_timer_comp = timer
                                    d_attention_comp = d_attention

                        else:
                            d_wrist_comp = 0
                            #speed_hand = 0
                            #speed_eye = 0
                        
                        # angle_wrist = math.atan( (int(wrist[1]*vheight) - y_center) / (int(wrist[0]*vwidth) - x_center) )
                        # angle_finger = math.atan( (int(wrist[1]*vheight) - y_center) / (int(wrist[0]*vwidth) - x_center) )

                        # Count 4 fingetip distance to fingertip tumb
                        d_attention = int (math.sqrt(pow(center_x-red_x,2) + pow(center_y-red_y,2)))
                        d_attention_list.append(d_attention)



                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Show Object Focused

                timer = round(time.time()-start_time,2)/2
                object_focus = min(d_attention_list, default="EMPTY")

                if object_focus != "EMPTY":
                    object_focus_id = d_attention_list.index(object_focus)

                    if (d_attention_list[ object_focus_id ] < (d_diagonal_list[ object_focus_id ]/2) ) :

                        str_object_focus = str(name_object_list[ object_focus_id ])
                        cv2.putText(frame, "Attention to " + str_object_focus, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

                        object_temp = str_object_focus

                        #print(str(timer) + " - Attention to " + str_object_focus)
                        if str_object_focus == "block": 
                            att_block = 2
                            att_task  = None
                            acc_block += 1

                        elif str_object_focus == "phone":
                            att_block = None
                            att_task  = 2
                            acc_phone += 1

                        else:
                            att_block = None
                            att_task  = None
                        

                    #else:
                    #    cv2.putText(frame, object_temp, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                    #    print(str(timer))
                #else:
                #    cv2.putText(frame, object_temp, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                #    print(str(timer))

                


                #------------------------------------------------ 

                if hand_status:
                                
                    # Interaction focus on attention
                    focus_x = red_x     #center_x
                    focus_y = red_y     #center_y 

                    # Draw 5 fingetip to attention
                    if red_x !=0 or red_y !=0 :                        
                        cv2.line(frame, (focus_x, focus_y), (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)), color=(0, 255, 0), thickness=2)
                        cv2.line(frame, (focus_x, focus_y), (int(tip_1[0]*vwidth), int(tip_1[1]*vheight)), color=(0, 200, 0), thickness=2)
                        cv2.line(frame, (focus_x, focus_y), (int(tip_2[0]*vwidth), int(tip_2[1]*vheight)), color=(0, 150, 0), thickness=2)
                        cv2.line(frame, (focus_x, focus_y), (int(tip_3[0]*vwidth), int(tip_3[1]*vheight)), color=(0, 100, 0), thickness=2)
                        cv2.line(frame, (focus_x, focus_y), (int(tip_4[0]*vwidth), int(tip_4[1]*vheight)), color=(0, 50, 0), thickness=2)

                    # Draw 4 fingetip to tumb-tip    
                    #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_1[0]*vwidth), int(tip_1[1]*vheight)), color=(0, 200, 0), thickness=2)
                    #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_2[0]*vwidth), int(tip_2[1]*vheight)), color=(0, 150, 0), thickness=2)
                    #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_3[0]*vwidth), int(tip_3[1]*vheight)), color=(0, 100, 0), thickness=2)
                    #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_4[0]*vwidth), int(tip_4[1]*vheight)), color=(0, 50, 0), thickness=2)

                    # Count fingetip distance to attention
                    tip_0[0] = int((tip_0[0] * vwidth) - focus_x)
                    tip_0[1] = int((tip_0[1] * vheight) - focus_y)
                    tip_1[0] = int((tip_1[0] * vwidth) - focus_x)
                    tip_1[1] = int((tip_1[1] * vheight) - focus_y)
                    tip_2[0] = int((tip_2[0] * vwidth) - focus_x)
                    tip_2[1] = int((tip_2[1] * vheight) - focus_y)
                    tip_3[0] = int((tip_3[0] * vwidth) - focus_x)
                    tip_3[1] = int((tip_3[1] * vheight) - focus_y)
                    tip_4[0] = int((tip_4[0] * vwidth) - focus_x)
                    tip_4[1] = int((tip_4[1] * vheight) - focus_y)

                    d_tip_0 = int (math.sqrt(pow(tip_0[0],2) + pow(tip_0[1],2)))
                    d_tip_1 = int (math.sqrt(pow(tip_1[0],2) + pow(tip_1[1],2)))
                    d_tip_2 = int (math.sqrt(pow(tip_2[0],2) + pow(tip_2[1],2)))
                    d_tip_3 = int (math.sqrt(pow(tip_3[0],2) + pow(tip_3[1],2)))
                    d_tip_4 = int (math.sqrt(pow(tip_4[0],2) + pow(tip_4[1],2)))

                    # Count 4 fingetip distance ke fingertip tumb
                    d_pinch_1 = int (math.sqrt(pow(tip_0[0]-tip_1[0],2) + pow(tip_0[1]-tip_1[1],2)))
                    d_pinch_2 = int (math.sqrt(pow(tip_0[0]-tip_2[0],2) + pow(tip_0[1]-tip_2[1],2)))
                    d_pinch_3 = int (math.sqrt(pow(tip_0[0]-tip_3[0],2) + pow(tip_0[1]-tip_3[1],2)))
                    d_pinch_4 = int (math.sqrt(pow(tip_0[0]-tip_4[0],2) + pow(tip_0[1]-tip_4[1],2)))

                    hand_status = False

                    #timer = round(time.time()-start_time,2)/2
                    hand_position = [d_tip_0, d_tip_1, d_tip_2, d_tip_3, d_tip_4, d_pinch_1, d_pinch_2, d_pinch_3, d_pinch_4]

                    #row_id = str(group_id) + "_" + str(measurement_id)

                    ### PREDICT ACTION

                    # Feature Scaling
                    coll_hand.append(hand_position)

                    if len(coll_hand) == sequence_data:
                        x_data = np.array(list(coll_hand))
                        x_train = sc_input.transform(x_data)
                        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
                        x_train = x_train[None, :, :]
                                
                        outputs = model_nn(x_train)
                        confidence, predicted = torch.max(outputs.data, 1)
                        
                        str_conf = str( format(confidence.item()*10,".2f") )

                        if predicted.item() == 0:
                            cv2.putText(frame, "grasp " + str_conf + "%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

                            if confidence.item()*10 > 60:
                                att_grasp = 1
                            else:
                                att_grasp = None
                        
                        elif predicted.item() == 1:
                            cv2.putText(frame, "reach " + str_conf + "%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted.item() == 2:
                            cv2.putText(frame, "release " + str_conf + "%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                        elif predicted.item() == 3:
                            cv2.putText(frame, "wonder " , (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2) #+ str_conf + "%"


                # Save to CSV    
                
                if (red_x != 0) and (red_y != 0):
                    writer_att.writerow([ timer, red_x, red_y, att_task,  att_block , att_grasp])
                    #print([ timer,  red_x, red_y, att_task,  att_block , att_grasp])


                print (str ([acc_block, acc_phone, acc_frame]))



                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Show Result

                cv2.imshow('Stream', frame ) #np.squeeze(results_detection.render())
                
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            n_frame = n_frame + 1

cap.release()
cv2.destroyAllWindows()

# Offline Testing

#img = r"C:\Users\anomt\Desktop\frame_06.jpg"
#results = model(img)
#results.print()
#cv2.imshow('Stream', np.squeeze(results.render()))
#cv2.waitKey(0)
#cv2.destroyAllWindows()