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
from IPython.display import display

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
    x_offset = 25
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
    


# Car information

# car = NvidiaRacecar()
# throttle_range = (0.27, 0.3)
# steering_range = (-0.99, 0.99)
# throttle_gain = 0.42
# throttle = 0.28
# steering_prev = 0
# throttle_control = 0





def line_tracking(diff, steering_range):
    global prev_steering
    print("diff : ",diff)
    curr_steering = - np.sign(diff) * abs((float(diff) * 4))**1.3 * steering_range[1] # steering_offset
    car.steering =  curr_steering * 0.7 + prev_steering * 0.3
    print("Steering : ",car.steering)
    prev_steering = - np.sign(diff) * abs((float(diff) * 4))**1.3 * steering_range[1]



        
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
diff = 0
prev_diff = 0
cnt = 0
curr_steering = 0
prev_steering = 0






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
x_ref = 319.5


os.environ["SDL_VIDEODRIVER"] = "dummy"

running = True
stop = False


# pth path
yolo_pth = "./pt/yolo_intersection.pt"
alexnet_pth = "./pt/best_fail.pth"
sign_dict = {0: 'bus_sign', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight'}

YOLO_model, colors_for_yolo, class_for_yolo = set_model_YOLO(yolo_pth)
ALEXNET_model = set_model_ALEXNET(alexnet_pth)



sensor_id = 0
downscale = 2
width, height = (1280, 720)
_width, _height = (width // downscale, height // downscale)
frame_rate = 15
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

streaming = True
if streaming == True : 
    cv2.namedWindow("Camera for model")



if cap.isOpened():
    try : 
        while running:
            
            
            pygame.event.pump()
            throttle = -joystick.get_axis(1)
            throttle = max(throttle_range[0], min(throttle_range[1], throttle))
            car.throttle = throttle
            
            
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            _, frame = cap.read()
            cv2.imshow("Original",frame)
            frame_for_cap = frame.copy()
            
            # Initialize cx
            # if cap.get(cv2.CAP_PROP_POS_FRAMES) < 10:
            #         # Lane Tracking
            
            #     frame_for_line = frame.copy()
            #     # frame_for_line = cv2.flip(frame_for_line,-1) # filp frame image
                
                
            #     # px_off = 13
            #     cropped_img = frame_for_line[210:320,140:467] # cropping image
            
            #     grayed = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
            
            #     blurred = cv2.GaussianBlur(grayed,(5,5),0)
            #     cv2.imshow("blurred",blurred)
            
            #     _,threshold = cv2.threshold(blurred,220,255,cv2.THRESH_BINARY_INV) # WB : 0~255 123기준으로 WB(White,Black)변환
            #     cv2.imshow("threshold",threshold)
                
            #     # Denoise
            #     mask = cv2.erode(threshold,None,iterations=2)
            #     mask = cv2.dilate(mask,None,iterations=2)
                
            #     contours,hierarchy = cv2.findContours(mask.copy(),1,cv2.CHAIN_APPROX_NONE)
            
            #     if len(contours) > 0:
            #         c = max(contours,key = cv2.contourArea)
            #         M = cv2.moments(c)
                    
            #         # Caculating CP
            #         cx = M['m10']/M['m00']
            #         x_ref = cx
            #         print("Reference Center X : ",cx)
                
            #     time.sleep(1)
                
            
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 8 == 7:
                
                pred_YOLO= YOLO_model(frame)
                
                if pred_YOLO[0].boxes.cls.nelement() == 1:
                    
                    pred_sign = sign_dict[pred_YOLO[0].boxes.cls.item()]  # predicted class (traffic sign)
                    draw_boxes(frame_for_cap, pred_YOLO, class_for_yolo, colors_for_yolo)
                    # sign_control(pred_YOLO, pred_sign)
            
            # Lane Tracking
           
            frame_for_line = frame.copy()
            
            # xpoint_off = 23
            # cropped_img = frame_for_line[270:360,115:492] # cropping image

            src_points = np.array([[206,270],[383,270],[433,315],[510,360],[79,360],[156,315]],dtype=np.float32)
            mask = np.zeros((frame_height,frame_width), dtype = np.uint8)
            cv2.fillPoly(mask, [src_points.astype(np.int32)],255)
            cropped_img = cv2.bitwise_and(frame_for_line,frame_for_line,mask=mask)
        
            grayed = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
          
            blurred = cv2.GaussianBlur(grayed,(5,5),0)
        
            _,threshold = cv2.threshold(blurred,200,255,cv2.THRESH_BINARY_INV) # WB : 0~255 200기준으로 WB(White,Black)변환
            cv2.imshow("threshold",threshold)
            
            # Denoise
            denoised = cv2.erode(threshold,None,iterations=2)
            denoised = cv2.dilate(denoised,None,iterations=2)
            
            contours,hierarchy = cv2.findContours(denoised.copy(),1,cv2.CHAIN_APPROX_NONE)

            if len(contours) > 0:
                c = max(contours,key = cv2.contourArea)
                M = cv2.moments(c)
                
                # Caculating CP
                cx = M['m10']/M['m00']

                print("Recent Center X : ",cx)
            if abs(diff - prev_diff) > 10 :
                diff = prev_diff
            else:   
                diff = (cx - x_ref) * 1.3 # Amplicate
                    # 초기화한 값 -0.4 기준으로 +0.42
                line_tracking(diff, steering_range)
                prev_diff = diff 
        
        
        
            if streaming == True:
                cv2.imshow("Camera for model", cropped_img)
                cv2.waitKey(1)
            
            if joystick.get_button(6):
                stream = True
                
            if stream == True:
                #cv2.imwrite('./images/record/{}.jpg'.format(timestamp),frame_for_cap)
                pass
            
                
            if joystick.get_button(11):
                cap.release()
                running = False
            
                 
                
            
            

    except Exception as e:
        running = False
        print(e)

    finally:
        cap.release()