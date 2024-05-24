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
import matplotlib
matplotlib.use('Agg')  # 터미널에서 실행할 때 필요한 백엔드 설정
import matplotlib.pyplot as plt
import pygame


def set_model_YOLO(yolo_pth) -> None:
    from ultralytics import YOLO
    YOLO_model = YOLO(yolo_pth, task='detect')
    classes_for_yolo = YOLO(yolo_pth, task='detect').names              
    colors_for_yolo = np.random.randn(len(classes_for_yolo), 3)
    colors_for_yolo = (colors_for_yolo * 255.0).astype(np.uint8)
    print("Done set yolo")
    return YOLO_model, colors_for_yolo, classes_for_yolo

def set_model_ALEXNET_left(alexnet_left_pth) -> None:
    
    ALEXNET_left_model = torchvision.models.alexnet(num_classes=4, dropout = 0.0).to("cuda")
    ALEXNET_left_model.load_state_dict(torch.load(alexnet_left_pth, map_location='cuda'))
    print("Done set alex")
    return ALEXNET_left_model


def set_model_ALEXNET_right(alexnet_right_pth) -> None:
    
    ALEXNET_right_model = torchvision.models.alexnet(num_classes=4, dropout = 0.0).to("cuda")
    ALEXNET_right_model.load_state_dict(torch.load(alexnet_right_pth, map_location='cuda'))
    print("Done set alex")
    return ALEXNET_right_model
   

def set_model_ALEXNET(alexnet_pth) -> None:
    
    ALEXNET_model = torchvision.models.alexnet(num_classes=4, dropout = 0.0).to("cuda")
    ALEXNET_model.load_state_dict(torch.load(alexnet_pth, map_location='cuda'))
    ALEXNET_model.eval()
    print("Done set alex")
    return ALEXNET_model


def output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height):
    
    pil_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    frame_for_alexnet = preprocess(pil_image)
    pred_ALEXNET = ALEXNET_model(frame_for_alexnet).detach().cpu().numpy()
    
    out_of_range, intersection, xn, yn = pred_ALEXNET[0]
    x_offset = 0
    xpre = ( xn.item() / 2 + 0.5 ) * frame_width + x_offset
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

car = NvidiaRacecar()
throttle_range = (0.27, 0.4)
steering_range = (-0.99, 0.99)
throttle_gain = 0.36
throttle = 0.37
steering_prev = 0
throttle_control = 0
car.throttle_gain = throttle_gain

right_count = 0
left_count = 0
bus = 0
cross_walk = 0

straight_flag = 0


def line_tracking(diff):
    global throttle, throttle_control
    # print("steering = ",- diff/150 * steering_range[1])
    car.steering = - np.sign(diff) * abs((diff/145)) ** 1.3 * steering_range[1]
    if abs(car.steering-steering_prev) > 0.2:
        throttle -= abs(car.steering-steering_prev)*0.006
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    else :
        throttle=0.37
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    
    print(car.throttle,car.throttle_gain)
    


def sign_control(pred_YOLO, pred_sign):
        global stop, count, running, steering_prev
        w,h = pred_YOLO[0].boxes.xywh[0][-2:]

        box_area = (w*h).item()
        

        ########################### plus condition " object dectection probability > 0.6 ""
        if box_area > 7000:
            if pred_sign == 'bus_sign' and count[0] == 0:
                vel_down = 0
                while vel_down > 0.1:
                    car.throttle -= 0.001
                    vel_down += 0.001
                count[0] += 1

            elif pred_sign == 'crosswalk' and count[1] == 0:
                stop =True
                while stop:
                    car.throttle -= 0.0001
                    if car.throttle <= 0 :
                        stop = False
                        break
                time.sleep(2)
                k = 0
                while k < throttle:
                    k += 0.0001
                    car.throttle = k
                count[1] += time.time()

            elif pred_sign == 'left' and count[2] == 0:
                out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_left_model, frame_width, frame_height)
                # draw_circles(frame, xpre, ypre)
                print("left sign")
                t_left = time.time()
                while time.time() - t_left < 2:
                    # 2초동안 교차로 left 모델 사용
                    diff_off = 0
                    diff = frame_width//2 - xpre + diff_off
                    
                    car.throttle = throttle
                    line_tracking(diff)
                    steering_prev = car.steering
                count[2] += time.time()
                
            elif pred_sign == 'right' and count[3] == 0:
                out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_right_model, frame_width, frame_height)
                print("right_model_loaded")
                # draw_circles(frame, xpre, ypre)
                print("right sign")
                
                t_right = time.time()
                while time.time() - t_right < 2:
                    # 2초동안 교차로 right 모델 사용
                    diff_off = 0
                    diff = frame_width//2 - xpre + diff_off

                    car.throttle = throttle
                    line_tracking(diff)
                    steering_prev = car.steering
                count[3] += time.time()
                
            elif pred_sign == 'straight' and count[4] == 0:
                count[4] += time.time()
                pass

                



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
alexnet_left_pth = "./pt/intersection_left_revised_best.pth"
alexnet_right_pth = "./pt/intersection_right_revised_best.pth"
sign_dict = {0: 'bus_sign', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight'}

YOLO_model, colors_for_yolo, class_for_yolo = set_model_YOLO(yolo_pth)
ALEXNET_model = set_model_ALEXNET(alexnet_pth)
ALEXNET_model.eval()
ALEXNET_left_model = set_model_ALEXNET_left(alexnet_left_pth)
ALEXNET_right_model = set_model_ALEXNET_right(alexnet_right_pth)

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

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

diff_values = []
filtered_diff_values = []

if cap.isOpened():
    stream = False
    index = 1
    try :
        # moving_avg_filter = MovingAverageFilter(window_size = 5) # 이동 평균 필터 초기화
        while running:
            pygame.event.pump()
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            _, frame = cap.read()
            frame_for_cap = frame.copy()
            
            # if False:
            #     pass
            
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 8 == 1:
                
                pred_YOLO= YOLO_model(frame)
                
                if pred_YOLO[0].boxes.cls.nelement() == 1:
                    
                    pred_sign = sign_dict[pred_YOLO[0].boxes.cls.item()]  # predicted class (traffic sign)
                    sign_control(pred_YOLO, pred_sign)
            

            out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height)
        
            diff_offset = 0
            diff = frame_width//2 - xpre - diff_offset
            print("diff : ",diff)

            # filtered_diff = moving_avg_filter.filter(diff)  # 필터링된 diff 값

            # #원래 diff 값과 필터링된 diff 값을 리스트에 추가
            # diff_values.append(diff)
            # filtered_diff_values.append(filtered_diff)
            if count[0] == 1:
                t_straight = time.time()
                while time.time() - t_straight < 2: # Line tracking during 2 seconds
                    car.throttle = throttle            
                    line_tracking(diff)
                    steering_prev = car.steering
                vel_plus = 0
                while vel_plus > 0.1:
                    car.throttle -= 0.001
                    vel_plus += 0.001
                count[0] += 1
            else: 
                car.throttle = throttle          
                line_tracking(diff)
                steering_prev = car.steering
                
            if stream == True :
                draw_circles(frame_for_cap,xpre,ypre)
                cv2.imwrite("./images/trial_2/{:05d}.jpg".format(index), frame_for_cap)
                index += 1
                
            if joystick.get_button(6):
                stream = True
            if joystick.get_button(11):
                car.throttle = 0
                car.steering = 0
                stop = True
                while stop:
                    car.throttle -= 0.0001
                    if car.throttle <= 0 :
                        stop = False
                        break
                running = False
                print("bus : ",count[0],"crosswalk : ",count[1],"left : ",count[2],"right : ",count[3],"stragiht : ",count[4])   
            

    except Exception as e:
        running = False
        print("bus : ",count[0],"crosswalk : ",count[1],"left : ",count[2],"right : ",count[3],"stragiht : ",count[4])
        print(e)

    finally:
        cap.release()
        
        # diff 값과 필터링된 diff 값을 그래프로 표시
        # plt.figure(figsize=(12, 6))
        # plt.plot(diff_values, label='Original diff values')
        # plt.plot(filtered_diff_values, label='Filtered diff values')
        # plt.xlabel('Frame')
        # plt.ylabel('Diff Value')
        # plt.title('Original and Filtered Diff Values')
        # plt.legend()
        # plt.savefig('diff_values_plot.png')  # 그래프를 파일로 저장
        # plt.close()  # 백엔드 오류를 방지하기 위해 그래프를 닫음
        # plt.show()