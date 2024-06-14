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
import Jetson.GPIO as GPIO
from model_setting import *


# 작업순서 : 두 개 이상의 객체를 인식하도록 코드 수정(O) -> 무단횡단(O) -> 신호등(O) -> 버스정류장(O) -> ACC(O)

####################### Setting for joystick #######################
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Constant for GPIO
TRIG = 16
ECHO = 22


####################### Initialize GPIO #######################
# Clean up any previous GPIO settings
GPIO.cleanup()  
if GPIO.getmode() is None:
    GPIO.setmode(GPIO.BOARD)

# Check if pins are valid
try:
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
except ValueError as e:
    print(f"GPIO setup error: {e}")
    
GPIO.output(TRIG, GPIO.LOW)

####################### Initialize car #######################
car = NvidiaRacecar()
car.throttle_gain = 0.313 # 0.34 # reset 0.315
throttle_range = (0.32, 0.60)
steering_range = (-0.99, 0.99)
throttle = 0.36
modified_throttle = 0.36
steering_prev = 0


####################### Initialize YOLO, ALEXNET #######################
yolo_pth = "./pt/YOLO_ver11.engine"
alexnet_pth = "./pt/ALEXNET_ver1.pth"
alexnet_left_pth = "./pt/merged_left_alexnet_epoch30.pth"
alexnet_right_pth = "./pt/merged_right_alexnet_epoch30.pth"
sign_dict = {0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pedestrian', 5: 'red', 6: 'right', 7 : 'straight', 8:'yellow'}
count_dict = {'bus_sign' : 0, 'crosswalk' : 0, 'green' : 0, 'left' : 0, 'pedestrian' : 0, 'red' : 0, 'right' : 0, 'straight':0, 'yellow' : 0}

YOLO_model, colors_for_yolo, class_for_yolo = set_model_YOLO(yolo_pth)

ALEXNET_model = set_model_ALEXNET_2(alexnet_pth)
ALEXNET_model.eval()
ALEXNET_left_model = set_model_ALEXNET(alexnet_left_pth)
ALEXNET_left_model.eval()
ALEXNET_right_model = set_model_ALEXNET(alexnet_right_pth)
ALEXNET_right_model.eval()


####################### Lane tracking functions #######################
def lane_tracking(diff):
    global throttle
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1] # 140
    if abs(car.steering - steering_prev) > 0.2:
        throttle -= abs(car.steering - steering_prev) * 0.005
    else:
        throttle = modified_throttle
    car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))

def lane_tracking_slowly(diff):
    global throttle
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1]
    if abs(car.steering - steering_prev) > 0.2:
        throttle -= abs(car.steering - steering_prev) * 0.007
    else:
        throttle = 0.343
    car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    
def lane_tracking_for_ACC(diff):
    global throttle_ACC
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1] # 140
    if abs(car.steering - steering_prev) > 0.2:
        throttle_ACC -= abs(car.steering - steering_prev) * 0.005
        
    car.throttle = max(throttle_range[0], min(throttle_range[1], throttle_ACC))



####################### ACC Controller #######################
alpha = 0.7  # EMA 필터의 평활화 계수
filtered_distance = None  # 필터링된 거리 초기화

def distance_check(): # 거리 계산 함수 
    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.00001)  # 10us
    GPIO.output(TRIG, GPIO.LOW)

    
    start = time.time()
    stop = time.time()  # stop 변수를 초기화
    
    start_time = time.time()
    while GPIO.input(ECHO) == GPIO.LOW:
        start = time.time()
        if start - start_time > 0.02:  # 20ms 후 타임아웃
            print("Timeout waiting for ECHO to go HIGH")
            return 60
        time.sleep(0.00001)

    start_time = time.time()
    while GPIO.input(ECHO) == GPIO.HIGH:
        stop = time.time()
        if stop - start_time > 0.02:  # 20ms 후 타임아웃
            print("Timeout waiting for ECHO to go LOW")
            return 60
        time.sleep(0.00001)

    duration = stop - start
    distance = (duration * 340 * 100) / 2  # 거리 계산
    return distance

def apply_ema_filter(new_distance):
    global filtered_distance
    if filtered_distance is None:
        filtered_distance = new_distance  # 첫 값을 초기화
    else:
        filtered_distance = alpha * new_distance + (1 - alpha) * filtered_distance
    return filtered_distance

Kp = 0.005  # 비례 이득
Ki = 0.0001  # 적분 이득
Kd = 0.0001  # 미분 이득

previous_error = 0
integral = 0

target_distance = 25  # 목표 거리 (cm)
emergency_distance = 5  # 급정거 거리 (cm)

def pid_control(current_distance, dt):
    global previous_error, integral

    error = -(target_distance - current_distance)
    integral += error * dt
    derivative = -(error - previous_error) / dt
    previous_error = error

    output = Kp * error + Ki * integral + Kd * derivative
    return output





####################### Car behavior rules #######################
# Variable for initializing bus, crosswalk, pedestrian
prev_bus = 0
prev_crosswalk = 0
prev_pedestrian = 0
traffic_mode = None

# 0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pesdestrian', 5: 'red', 6: 'right', 7:'yellow'
# size : bus_sign / crosswalk, left, right/ red, yellow, green / pedestrian
def sign_control(num_objects, classes = None, bounding_boxes = None, confidences = None):
    global x1_ped, x2_ped, frame_cnt_traffic_light, whether_stop, prev_bus, prev_crosswalk, prev_pedestrian, traffic_mode
    

    prev_bus = count_dict['bus_sign']
    prev_crosswalk = count_dict['crosswalk']
    prev_pedestrian = count_dict['pedestrian']

    if num_objects > 0:
        for i in range(num_objects):
            
            class_id = classes[i]
            # Count for each object
            pred_sign = sign_dict[class_id]
            confidence = confidences[i]
            box = bounding_boxes[i]
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            box_area = w * h
        
            # Detection for bus_sign
            if box_area > 1500 and confidence > 0.5 and pred_sign == 'bus_sign':
                traffic_mode = 'bus mode'
                count_dict['bus_sign'] += 1
                    
            # Detection for traffic sign(expect bus)
            if box_area > 3000 and confidence > 0.65: #### 3000 or 4500
                if pred_sign == 'crosswalk':
                    count_dict['crosswalk'] += 1
                    traffic_mode = "crosswalk mode"
                elif pred_sign == 'left' and count_dict['left'] == 0:
                    count_dict['left'] += 1
                    count_dict['right'] = 0
                    traffic_mode = "left intersection mode"
                elif pred_sign == 'right' and count_dict['right'] == 0:
                    count_dict['right'] += 1
                    count_dict['left'] = 0
                    traffic_mode = 'right intersection mode'

            # Detection for pedestrian
            if box_area > 3000  and confidence > 0.4: # 보행자 인지시 급정거
                if pred_sign == 'pedestrian':
                    count_dict['pedestrian'] += 1
                    if box_area > 10000:
                        x1_ped = x1
                        x2_ped = x2
                        whether_stop += 1
                        traffic_mode = 'pedestrian stopping mode'
                    
            # Detection for traffic light
            if y1 < 23 and confidence > 0.5: # 신호등 bouinding box의 y좌표를 활용하여 적절한 거리에서 counting
                if pred_sign == 'red' and count_dict['red'] == 0  and 5 < y1:
                    count_dict['green'] = 0
                    count_dict['red'] += 1
                    traffic_mode = 'red mode'
                elif pred_sign == 'yellow' and count_dict['yellow'] == 0 and 5 < y1:
                    count_dict['green'] = 0
                    count_dict['yellow'] += 1
                    traffic_mode = 'yellow mode'
                elif pred_sign == 'green' and count_dict['green'] == 0:
                    count_dict['green'] += 1
                    count_dict['red'] = 0
                    count_dict['yellow'] = 0
                    traffic_mode = 'green mode'
                
    if prev_pedestrian == count_dict['pedestrian']:
            count_dict['pedestrian'] = 0
            whether_stop = 0
    if prev_crosswalk == count_dict['crosswalk']:
        count_dict['crosswalk'] = 0
    if prev_bus == count_dict['bus_sign']:
        count_dict['bus_sign'] = 0

####################### Initialize camera #######################
sensor_id = 0
downscale = 2
width, height = 1280, 720
_width, _height = (width // downscale, height // downscale) # frame size : 360,640
frame_width = width // downscale
frame_height = height // downscale
frame_rate = 18
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
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)


####################### Main loop initialization #######################
# Initialize flag values 0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pesdestrian', 5: 'red', 6: 'right', 7:'yellow'
count = [0, 0, 0, 0, 0, 0, 0, 0]
whether_stop = 0
flag_bus = 0

# Initialize frame count for rules
frame_cnt_bus = 0
frame_cnt_bus2 = 0
frame_cnt_left = 0
frame_cnt_right = 0
frame_cnt_cross = 0
frame_cnt_pedestrian = 0
frame_cnt_traffic_light = 0

# Temporary variable for saving pesdestrian info
x1_ped = 9999
x2_ped = 9999 # boundary 사이에 들어오지 않는 큰 값으로 설정(쓰레기값)


# boundary1,2 help to decide whether pedestrian is in lane
boundary1 = 130
boundary2 = 510 

intersection = 0
running = True
save_image = True
stream = False
image_index = 1


if save_image == True:
    save_path = './images'
    os.makedirs(save_path,exist_ok=True)
    save_path = save_path +'/'+ f'{len(os.listdir(save_path))+1:06d}'
    os.makedirs(save_path,exist_ok=True)



###################################################################################################################################################################################
#################################################################################### Main loop ####################################################################################
###################################################################################################################################################################################

if cap.isOpened():
    try:
        previous_time = time.time()
        while running:
            pygame.event.pump()
            _, frame = cap.read()
            frame_for_cap = frame.copy()
            traffic_mode = None
            # Object detection with YOLO
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 1:
                pred_YOLO = YOLO_model(frame, verbose = False)
                num_objects = pred_YOLO[0].boxes.cls.nelement()
                classes = None
                bounding_boxes = None
                confidences = None
                if num_objects > 0:
                    bounding_boxes = pred_YOLO[0].boxes.xyxy.cpu().numpy()
                    classes = pred_YOLO[0].boxes.cls.cpu().numpy()
                    confidences = pred_YOLO[0].boxes.conf.cpu().numpy()
                    draw_boxes(frame_for_cap, pred_YOLO[0], class_for_yolo, colors_for_yolo)

                sign_control(num_objects, classes, bounding_boxes, confidences)
            
# Initialize flag values 0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pesdestrian', 5: 'red', 6: 'right', 7:'yellow'
            
            # [Priority] no collision with pedestrian > no breaking the traffic law(traffic light) > Other traffic sign
            # Priority1 : check whether pedestrain is detected
            if whether_stop > 0 and ((boundary1 < x1_ped < boundary2) or (boundary1 < x2_ped < boundary2)) and frame_cnt_pedestrian < frame_rate * 3:
                    # 급정거
                    car.throttle = 0
                    frame_cnt_pedestrian += 1
                    print("Stop for pedestrian")

            else:
                # Initialize cnt for pedestrain
                frame_cnt_pedestrain = 0
                x1_ped = 9999
                x2_ped = 9999
                
                # Set appropriate lane tracking model 
                time_for_left_right = 2.3 # time for changing alexnet model temporarily
                if count_dict['left'] == 1 and frame_cnt_left < frame_rate * time_for_left_right:
                    mode = 'left'
                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_left_model, frame_width, frame_height)
                    print("Left on")
                    count_dict['right'] = 0
                    if not (count_dict['red'] or count_dict['yellow']):
                        frame_cnt_left += 1
                elif count_dict['right'] == 1 and frame_cnt_right < frame_rate * time_for_left_right:
                    mode = 'right'
                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_right_model, frame_width, frame_height)
                    print("Right on")
                    count_dict['left'] = 0
                    if not (count_dict['red'] or count_dict['yellow']):
                        frame_cnt_right += 1

                else:
                    mode = 'straight'
                    # Initialize count for left, right sign
                    count_dict['left'] = 0
                    count_dict['right'] = 0
                    # Initialize frame count left, right
                    if frame_cnt_left >= frame_rate * time_for_left_right:
                        frame_cnt_left = 0
                    if frame_cnt_right >= frame_rate * time_for_left_right:
                        frame_cnt_right = 0

                    # out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height)
                    xpre = output_model_ALEXNET_2(frame, ALEXNET_model, frame_width, frame_height)
                # Calculate difference for controller
                diff = frame_width // 2 - xpre
                
                # Priority2 : Check for need to stop(red or yellow light detection)
                need_to_stop = count_dict['red'] or count_dict['yellow']
                if need_to_stop == 1:
                    if frame_cnt_traffic_light < frame_rate * 1.8:
                        car.throttle -= 0.02
                        frame_cnt_traffic_light += 1
                        if car.throttle < 0:
                            car.throttle = 0
                    else:
                        car.throttle = 0
                    print("Stop at traffic light")
                
                        
                else: # Priority3 : green light and other traffic sign
                    frame_cnt_traffic_light = 0
                    if count_dict['bus_sign'] > 0 and count_dict['pedestrian'] > 0 :
                        flag_bus = 1
                    if count_dict['crosswalk'] > 0 and frame_cnt_cross < frame_rate * 3: # crosswalk
                        car.throttle = 0
                        print("Stop")
                        frame_cnt_cross += 1
                    elif flag_bus == 1 and frame_cnt_bus < frame_rate * 4: # bus_sign, pedestrian
                        if frame_cnt_bus2 < frame_rate * 1.25:
                            lane_tracking_slowly(diff)
                            frame_cnt_bus2 += 1
                            frame_cnt_bus += 1
                        else:
                            car.throttle = 0
                            print("Stop for bus passenger")
                            steering_prev = car.steering
                            frame_cnt_bus += 1

                    else: # Normal tracking mode without any object or traffic sign
                        result_distance = distance_check()
                        
                        if result_distance is not None:
                            filtered_result = apply_ema_filter(result_distance)
                            # print("distance = {:.2f} cm, filtered distance = {:.2f} cm".format(result_distance, filtered_result))

                            current_time = time.time()
                            dt = current_time - previous_time
                            previous_time = current_time

                            if filtered_result < emergency_distance: # Stop short
                                car.throttle = 0
                                print("Emergency stop")
                            elif filtered_result > target_distance: # ACC off
                                lane_tracking(diff)
                                steering_prev = car.steering
                            else : # ACC on
                                # PID controller -> throttle control
                                throttle_adjustment = pid_control(filtered_result, dt)
                                throttle_ACC = max(throttle_range[0], min(throttle_range[1], throttle + throttle_adjustment))
                                lane_tracking_for_ACC(diff)
                                steering_prev = car.steering
                                print(f"Adjusting speed: throttle = {car.throttle}")
                            
               
                        

            # For checking whether object detection is well done
            print(count_dict)

            if save_image:
                draw_circles(frame_for_cap, xpre, 275, mode, traffic_mode)
                draw_boxes(frame_for_cap, pred_YOLO, class_for_yolo, colors_for_yolo)
                temp = str(save_path +'/'+ "{:05d}.jpg".format(image_index))
                cv2.imwrite(str(save_path +'/'+ "{:05d}.jpg".format(image_index)), frame_for_cap)
                image_index += 1
                
            if stream:
                draw_circles(frame_for_cap, xpre, 275, mode, traffic_mode)
                draw_boxes(frame_for_cap, pred_YOLO, class_for_yolo, colors_for_yolo)
                cv2.imshow("Camera with model", frame_for_cap)
                cv2.waitKey(1)
            
            if joystick.get_button(6):
                modified_throttle += 0.001
                print("ACCELERATED!!")
            if joystick.get_button(7):
                modified_throttle -= 0.001
                print("DECCELERATED !!")
            if joystick.get_button(11):
                car.throttle = 0
                car.steering = 0
                running = False
            
            print(f"throttle : {modified_throttle}")
            
###################################################################################################################################################################################
###################################################################################################################################################################################
###################################################################################################################################################################################

    except Exception as e:
        running = False
        cap.release()
        print(e)
    finally:
        cap.release()
