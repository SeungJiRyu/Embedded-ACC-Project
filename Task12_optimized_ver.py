import os
import cv2
from lib.jetracer.nvidia_racecar import NvidiaRacecar
import numpy as np
import time
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import pygame
import logging


os.environ["SDL_VIDEODRIVER"] = "dummy"

# Set model functions
def set_model_YOLO(yolo_pth):
    from ultralytics import YOLO
    YOLO_model = YOLO(yolo_pth, task='detect')
    classes_for_yolo = YOLO_model.names              
    colors_for_yolo = (np.random.randn(len(classes_for_yolo), 3) * 255.0).astype(np.uint8)
    return YOLO_model, colors_for_yolo, classes_for_yolo

def set_model_ALEXNET(pth):
    model = torchvision.models.alexnet(num_classes=4, dropout=0.0).to("cuda")
    model.load_state_dict(torch.load(pth, map_location='cuda'))
    model.eval()
    return model

def output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height):
    pil_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    frame_for_alexnet = preprocess(pil_image)
    pred_ALEXNET = ALEXNET_model(frame_for_alexnet).detach().cpu().numpy()
    out_of_range, intersection, xn, yn = pred_ALEXNET[0]
    x_offset = 0
    xpre = ( xn.item() / 2 + 0.5 ) * frame_width + x_offset
    ypre = ( yn.item() / 2 + 0.5 ) * frame_height
    return out_of_range, intersection, xpre, ypre

def draw_boxes(image, pred, classes, colors):
    for r in pred:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = colors[int(box.cls[0])].tolist()
            label = f"{classes[int(box.cls[0])]} {round(float(box.conf[0]), 2)}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_circles(image, xpre, ypre):
    cv2.circle(image, (int(xpre), int(ypre)), radius=5, color=(255, 0, 0), thickness=-1)

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(image):
    return TEST_TRANSFORMS(image).unsqueeze(0).to(torch.device('cuda'))

# Initialize car
car = NvidiaRacecar()
car.throttle_gain = 0.315 # 0.34 # reset 0.315
throttle_range = (0.32, 0.38)
steering_range = (-0.99, 0.99)
throttle = 0.37
steering_prev = 0

def line_tracking(diff):
    global throttle
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1] # 140
    if abs(car.steering - steering_prev) > 0.2:
        throttle -= abs(car.steering - steering_prev) * 0.005
    else:
        throttle = 0.37
    car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))

def line_tracking_slowly(diff):
    global throttle
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1]
    if abs(car.steering - steering_prev) > 0.2:
        throttle -= abs(car.steering - steering_prev) * 0.007
    else:
        throttle = 0.343
    car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))

def sign_control(pred_YOLO, pred_sign, frame, ALEXNET_left_model, ALEXNET_right_model, frame_width, frame_height):
    global throttle, steering_prev
    w, h = pred_YOLO[0].boxes.xywh[0][-2:]
    box_area = (w * h).item()

    if box_area > 4500 and float(pred_YOLO[0].boxes.conf) > 0.5: #####################################################
        if pred_sign == 'bus_sign' and count[0] == 0:
            count[0] += 1
        elif pred_sign == 'crosswalk' and count[1] == 0:
            count[1] += 1

    if box_area > 3000 and float(pred_YOLO[0].boxes.conf) > 0.5: ####### 3000 and 4500
        if pred_sign == 'left' and count[2] == 0:
            count[2] += 1
        elif pred_sign == 'right' and count[3] == 0:
            count[3] += 1
        elif pred_sign == 'straight' and count[4] == 0:
            count[4] += 1

# Initialize flag values
count = [0, 0, 0, 0, 0]
running = True

# Model paths
yolo_pth = "./pt/yolo_intersection.pt"
alexnet_pth = "./pt/version4_alexnet_2.pth"
alexnet_left_pth = "./pt/merged_left_alexnet_epoch30.pth"
alexnet_right_pth = "./pt/merged_right_alexnet_epoch30.pth"
sign_dict = {0: 'bus_sign', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight'}

YOLO_model, colors_for_yolo, class_for_yolo = set_model_YOLO(yolo_pth)
# YOLO_model.eval()
ALEXNET_model = set_model_ALEXNET(alexnet_pth)
ALEXNET_model.eval()
ALEXNET_left_model = set_model_ALEXNET(alexnet_left_pth)
ALEXNET_left_model.eval()
ALEXNET_right_model = set_model_ALEXNET(alexnet_right_pth)
ALEXNET_right_model.eval()

# Camera settings
sensor_id = 0
downscale = 2 ############################################
width, height = 1280, 720
_width, _height = (width // downscale, height // downscale)
frame_rate = 12 ##########################################
frame_cnt_bus = 0
frame_cnt_left = 0
frame_cnt_right = 0
frame_cnt_cross = 0


gstreamer_pipeline = (
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
    % (
        sensor_id,
        width,
        height,
        frame_rate,
        _width,
        _height,
    )
)




frame_width = width // downscale
frame_height = height // downscale

cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
intersection = 0
if cap.isOpened():
    stream = False
    index = 1
    try:
        while running:
            pygame.event.pump()
            _, frame = cap.read()
            frame_for_cap = frame.copy()

            # Yolo
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 4 == 1:
                pred_YOLO = YOLO_model(frame, verbose = False)
                if pred_YOLO[0].boxes.cls.nelement() == 1:
                    pred_sign = sign_dict[pred_YOLO[0].boxes.cls.item()]
                    print(pred_sign,pred_YOLO[0].boxes.conf)
                    sign_control(pred_YOLO, pred_sign, frame, ALEXNET_left_model, ALEXNET_right_model, frame_width, frame_height)
            
            # Controller
            time_for_left_right = 3 #################
            if count[2] == 1 and frame_cnt_left < frame_rate * time_for_left_right:
                
                out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_left_model, frame_width, frame_height)
                frame_cnt_left += 1
            elif count[3] == 1 and frame_cnt_right < frame_rate * time_for_left_right:
                out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_right_model, frame_width, frame_height)
                frame_cnt_right += 1
            else:
                out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height)
            

            diff = frame_width // 2 - xpre
            # print(f"out of range {out_of_range}, intersection : {intersection}")


            if count[1] == 1 and frame_cnt_cross < frame_rate * 3:
                car.throttle = 0
                print("Stop")
                frame_cnt_cross += 1
            elif count[0] == 1 and frame_cnt_bus < frame_rate * 2.2:
                line_tracking_slowly(diff)
                print("Decelelrated")
                steering_prev = car.steering
                frame_cnt_bus += 1
            else:
                line_tracking(diff)
                steering_prev = car.steering

            print("bus count : ",frame_cnt_bus," oss count : ",frame_cnt_cross," left count : ",frame_cnt_left," right count : ",frame_cnt_right)

            print(diff)

            if stream:
                draw_circles(frame_for_cap, xpre, ypre)
                cv2.imwrite(f"./images/trial_2/{index:05d}.jpg", frame_for_cap)
                index += 1
            if joystick.get_button(6):
                stream = True
            if joystick.get_button(11):
                car.throttle = 0
                car.steering = 0
                running = False
    except Exception as e:
        running = False
        cap.release()
        print(e)
    finally:
        cap.release()
        print(count)