from __future__ import annotations
import os
import cv2
from lib.jetracer.nvidia_racecar import NvidiaRacecar
from lib.jetcam.utils import bgr8_to_jpeg
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import datetime
import pygame
import matplotlib
matplotlib.use('Agg')  # 터미널에서 실행할 때 필요한 백엔드 설정
import matplotlib.pyplot as plt

os.environ["SDL_VIDEODRIVER"] = "dummy"


def set_model_YOLO(yolo_pth) -> None:
    from ultralytics import YOLO
    YOLO_model = YOLO(yolo_pth, task='detect')
    classes_for_yolo = YOLO(yolo_pth, task='detect').names              
    colors_for_yolo = np.random.randn(len(classes_for_yolo), 3)
    colors_for_yolo = (colors_for_yolo * 255.0).astype(np.uint8)
    print("Done set yolo")
    return YOLO_model, colors_for_yolo, classes_for_yolo
    

def set_model_ALEXNET(alexnet_pth) -> None:
    
    ALEXNET_model = torchvision.models.alexnet(num_classes=4, dropout = 0.0).to("cuda")
    ALEXNET_model.load_state_dict(torch.load(alexnet_pth, map_location='cuda'))
    print("Done set alex")
    return ALEXNET_model


def output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height):
    
    pil_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    frame_for_alexnet = preprocess(pil_image)
    pred_ALEXNET = ALEXNET_model(frame_for_alexnet).detach().cpu().numpy()
    
    out_of_range, intersection, xn, yn = pred_ALEXNET[0]
    x_offset = 0
    xpre = ( xn.item() / 2 + 0.5 ) * frame_width + x_offset
    print("xpre : ",xpre)
    ypre = ( yn.item() / 2 + 0.5 ) * frame_height
    return out_of_range, intersection, xpre, ypre


# Yolo function
def draw_boxes(image, pred, classes, colors_for_yolo):

    for r in pred:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = round(float(box.conf[0]), 2)
            label = int(box.cls[0])

            color = colors_for_yolo[label].tolist()
            cls_name = classes[label]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{cls_name} {score}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)



def draw_circles(image,xpre,ypre):
    cv2.circle(image, (int(xpre), int(ypre)), radius=5, color=(255, 0, 0),thickness=-1)



TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((640, 360)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(image: Image):
    device = torch.device('cuda')    
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]


class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def filter(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)


# Car information

# car = NvidiaRacecar()
# throttle_range = (0.27, 0.3)
# steering_range = (-0.99, 0.99)
# throttle_gain = 0.42
# throttle = 0.28
# steering_prev = 0
# throttle_control = 0





def line_tracking(diff, steering_range):

    car.steering = - np.sign(diff) * abs((diff/150)) ** 1.2 * steering_range[1]



        
# def line_tracking_slowly(diff):
#     car.steering = - diff/150 * steering_range[1]
#     car.throttle = 0.02 # 초기화한 값 -0.4 기준으로 +0.4


# def go_straight():
#     car.steering  = 0
#     car. throttle = throttle


# def turn_left():
#     car.steering  = 0.8
#     car. throttle = throttle


# def turn_right():
#     car.steering  = -0.8
#     car. throttle = throttle


# def sign_control(pred_YOLO, pred_sign):
#         global stop, count, running

#         w,h = pred_YOLO[0].boxes.xywh[0][-2:]

#         box_area = (w*h).item()
        

#         ########################### plus condition " object dectection probability > 0.6 ""
#         if box_area > 7000:
#             if pred_sign == 'bus_sign' and count[0] == 0:
#                 vel_plus = 0
#                 vel_down = 0
#                 while vel_down > 0.015:
#                     car.throttle -= 0.0001
#                     vel_down += 0.0001
#                 time.sleep(2)
#                 while vel_plus > 0.015:
#                     car.throttle -= 0.0001
#                     vel_plus += 0.0001
#                 count[0] += 1

#             elif pred_sign == 'crosswalk' and count[1] == 0:
#                 stop =True
#                 while stop:
#                     car.throttle -= 0.0001
#                     if car.throttle <= 0 :
#                         stop = False
#                         break
#                 k = 0
#                 while k < throttle:
#                     k += 0.0001
#                     car.throttle = k
#                 count[1] += 1
#                 time.sleep(2)

#             elif pred_sign == 'left':
#                 car.steering = -steering_range[0]
#                 pass
                
#             elif pred_sign == 'right':
#                 car.steering = steering_range[0]
#                 pass
                
#             elif pred_sign == 'straight':
#                 pass

                

## Car
car = NvidiaRacecar()
throttle_range = (-0.4, 0.4)
steering_range = (-1, 1)
car.throttle_gain = 0.34






# Initialize flag values for line tracking
bus_flag = 0
crossing_flag = 0
straight_flag = 0
left_flag = 0
right_flag = 0
object_detected = 0
count = [0,0,0,0,0]
bus_threshold = 0.2
bouding_box_area_threshold = 50 


os.environ["SDL_VIDEODRIVER"] = "dummy"

running = True
stop = False


# pth path
yolo_pth = "./pt/yolo_intersection.pt"
alexnet_pth = "./pt/version4_alexnet_2.pth"
sign_dict = {0: 'bus_sign', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight'}

YOLO_model, colors_for_yolo, class_for_yolo = set_model_YOLO(yolo_pth)
ALEXNET_model = set_model_ALEXNET(alexnet_pth)
ALEXNET_model.eval()



sensor_id = 0
downscale = 2
width, height = (1280, 720)
_width, _height = (width // downscale, height // downscale)
frame_rate = 26
flip_method = 0
gstreamer_pipeline = (
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
    "nvvidconv flip-method=%d ! "
    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
    % (
        sensor_id,
        width,
        height,
        frame_rate,
        flip_method,
        _width,
        _height,
    )
)
frame_width = 640
frame_height = 360
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
stream = False


pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

# streaming = True
# if streaming == True : 
#     cv2.namedWindow("Camera for model")

diff_values = []
filtered_diff_values = []

if cap.isOpened():
    try :
        #moving_avg_filter = MovingAverageFilter(window_size = 5) # 이동 평균 필터 초기화 
        while running:
            
            
            pygame.event.pump()
            throttle = -joystick.get_axis(1)
            throttle = max(throttle_range[0], min(throttle_range[1], throttle))
            car.throttle = throttle  

            
            
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            _, frame = cap.read()
            frame_for_cap = frame.copy()
            
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 1:
                
                pred_YOLO= YOLO_model(frame)
                
                if pred_YOLO[0].boxes.cls.nelement() == 1:
                    
                    pred_sign = sign_dict[pred_YOLO[0].boxes.cls.item()]  # predicted class (traffic sign)
                    draw_boxes(frame_for_cap, pred_YOLO, class_for_yolo, colors_for_yolo)
                    # sign_control(pred_YOLO, pred_sign)
            
            
            # Lane Tracking
            out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height)
            draw_circles(frame_for_cap, xpre, ypre)

            diff_off = 0
            diff = frame_width//2 - xpre + diff_off
            print("diff : ",diff)
            # filtered_diff = moving_avg_filter.filter(diff)  # 필터링된 diff 값

            #     # 원래 diff 값과 필터링된 diff 값을 리스트에 추가
            # diff_values.append(diff)
            # filtered_diff_values.append(filtered_diff)
                
            
            line_tracking(diff, steering_range)
                
            # if streaming == True:
            #     cv2.imshow("Camera for model", frame_for_cap)
            #     cv2.waitKey(1)
            
            if joystick.get_button(6):
                stream = True
                
            if stream == True:
                cv2.imwrite('./images/new_version4_alexnet_2/{}.jpg'.format(timestamp),frame)
            
                
            if joystick.get_button(11):
                car.throttle = 0
                stop = True
                while stop:
                    car.throttle -= 0.0001
                    if car.throttle <= 0 :
                        stop = False
                        break
                running = False
            
    except Exception as e:
        running = False
        print(e)

    finally:
        cap.release()

        plt.figure(figsize=(12, 6))
        plt.plot(diff_values, label='Original diff values')
        plt.plot(filtered_diff_values, label='Filtered diff values')
        plt.xlabel('Frame')
        plt.ylabel('Diff Value')
        plt.title('Original and Filtered Diff Values')
        plt.legend()
        plt.savefig('diff_values_plot.png')  # 그래프를 파일로 저장
        plt.close()  # 백엔드 오류를 방지하기 위해 그래프를 닫음
        plt.show()
