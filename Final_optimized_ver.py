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



# 작업순서 : 두 개 이상의 객체를 인식하도록 코드 수정(O) -> 무단횡단(O) -> 신호등(O) -> 버스정류장(O) -> ACC(O)

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

def lane_tracking(diff):
    global throttle
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1] # 140
    if abs(car.steering - steering_prev) > 0.2:
        throttle -= abs(car.steering - steering_prev) * 0.005
    else:
        throttle = 0.37
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
    global throttle
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1] # 140
    if abs(car.steering - steering_prev) > 0.2:
        throttle -= abs(car.steering - steering_prev) * 0.005
    else:
        throttle = 0.37
    car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))

def sign_control(pred_sign, box, confidence):
    global x1_pes, x2_pes
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    box_area = w * h

    if box_area > 4500 and confidence > 0.5: 
        if pred_sign == 'bus_sign' and count[0] == 0:
            count[0] += 1
        elif pred_sign == 'crosswalk' and count[1] == 0:
            count[1] += 1

    if box_area > 3000 and confidence > 0.5: #### 3000 or 4500
        if pred_sign == 'left' and count[2] == 0:
            count[2] += 1
        elif pred_sign == 'right' and count[3] == 0:
            count[3] += 1
        elif pred_sign == 'straight' and count[4] == 0:
            count[4] += 1
            
    if box_area > 5000 and confidence > 0.5: ###### 보행자를 보았을 때 멈추기 위한 적정 box_area 설정 필요
        if pred_sign == 'pedestrain':
            x1_pes = x1
            x2_pes = x2
            count[5] += 1
            
def sign_control_for_traffic_light(pred_sign, box, confidence):
    global frame_cnt_traffic_light
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    box_area = w * h

    if box_area > 2500 and confidence > 0.5: ########## 신호등 앞에서 적절하게 멈추도록 box_area 설정 필요 : 멀리서 인식하게 한 후, 2초에 걸쳐서 속도를 늦추고 정지하는 방법이 좋을듯
        if pred_sign == 'red_light' and count_for_traffic_light[0] == 0:
            count_for_traffic_light[0] += 1
        elif pred_sign == 'yellow_light' and count_for_traffic_light[1] == 0:
            # Initialize cnt and frame cnt for traffic light(green)
            count_for_traffic_light[2] = 0
            count_for_traffic_light[1] += 1
        else:
            count_for_traffic_light[2] += 1
            # Initialize cnt and frame cnt for traffic light(red, yellow)
            count_for_traffic_light[0] = 0
            count_for_traffic_light[1] = 0
            frame_cnt_traffic_light = 0



# Initialize flag values : bus_sign, crosswalk, left, right, straight, pedestrain
count = [0, 0, 0, 0, 0, 0]
# Initialize flag values : red_light, yellow_light, green_light
count_for_traffic_light = [0, 0, 0]

running = True

# Model paths
yolo_pth = "./pt/yolo_intersection.pt"
alexnet_pth = "./pt/version4_alexnet_2.pth"
alexnet_left_pth = "./pt/merged_left_alexnet_epoch30.pth"
alexnet_right_pth = "./pt/merged_right_alexnet_epoch30.pth"
sign_dict = {0: 'bus_sign', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight', 5: 'pedestrian'}
sign_dict_for_traffic_light = {0: 'red_light', 1: 'yellow_light', 2: 'green_right'}

YOLO_model, colors_for_yolo, class_for_yolo = set_model_YOLO(yolo_pth)
########p########################### Need to plus YOLO model for traffic light #############################
# YOLO_traffic_light_model, colors_for_yolo, class_for_yolo = set_model_YOLO(yolo_pth)


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
frame_cnt_pedestrian = 0
frame_cnt_traffic_light = 0

# Temporary variable for saving pesdestrian info
x1_pes = 9999
x2_pes = 9999 # boundary 사이에 들어오지 않는 큰 값으로 설정(쓰레기값)

# boundary1,2 can explain pedestrian is in lane
boundary1 = 2000
boundary2 = 3000 #### 수정 필요


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

            # Object detection with YOLO
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 4 == 1:
                pred_YOLO = YOLO_model(frame, verbose = False)
                num_objects = pred_YOLO[0].boxes.cls.nelement()
                
                if num_objects > 0:
                    bounding_boxes = pred_YOLO[0].boxes.xyxy.cpu().numpy()
                    classes = pred_YOLO[0].boxes.cls.cpu().numpy()
                    confidences = pred_YOLO[0].boxes.conf.cpu().numpy()
                    
                    for i in range(num_objects):
                        class_id = classes[i]
                        # Count for each object
                        pred_sign = sign_dict[class_id]
                        confidence = confidences[i]
                        box = bounding_boxes[i]
                        sign_control(pred_sign, box, confidence)
            
            # Traffic light detection with YOLO
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 4 == 3:
                pred_YOLO = YOLO_traffic_light_model(frame, verbose = False)
                num_objects = pred_YOLO[0].boxes.cls.nelement()
                
                if num_objects > 0:
                    bounding_boxes = pred_YOLO[0].boxes.xyxy.cpu().numpy()
                    classes = pred_YOLO[0].boxes.cls.cpu().numpy()
                    confidences = pred_YOLO[0].boxes.conf.cpu().numpy()
                    
                    for i in range(num_objects):
                        class_id = classes[i]
                        # Count for each object
                        pred_sign = sign_dict_for_traffic_light[class_id]
                        confidence = confidences[i]
                        box = bounding_boxes[i]
                        sign_control_for_traffic_light(pred_sign, box, confidence)
            
            
            # [Priority] no collision with pedestrian > no breaking the traffic law(traffic light) >  ...
            # Priority1 : check whether pedestrain is detected
            # 바운딩 박스가 차선 안에 들어와 있으면(= 충돌이 예상되면) 정지, 일단은 3초동안 일시 정지하도록 구현함 (일시 정지 시간을 없애봐도 괜찮을 듯)
            if count[5] >= 1 and ((boundary1 < x1_pes < boundary2) or (boundary1 < x2_pes < boundary2)) and frame_cnt_pedestrian < frame_rate * 3:
                    # 급정거
                    car.throttle = 0
                    frame_cnt_pedestrian += 1
                    print("Stop for pedestrian")
            else:
                # Initialize cnt for pedestrain
                frame_cnt_pedestrain = 0
                x1_pes = 9999
                x2_pes = 9999
                
                # Set appropriate lane tracking model 
                time_for_left_right = 3 # time for changing alexnet model temporarily
                if count[2] == 1 and frame_cnt_left < frame_rate * time_for_left_right:
                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_left_model, frame_width, frame_height)
                    frame_cnt_left += 1
                elif count[3] == 1 and frame_cnt_right < frame_rate * time_for_left_right:
                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_right_model, frame_width, frame_height)
                    frame_cnt_right += 1
                else:
                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height)
                
                # Calculate difference for controller
                diff = frame_width // 2 - xpre
                
                # Priority2 : Check for need to stop(red or yellow light detection)
                need_to_stop = count_for_traffic_light[0] or count_for_traffic_light[1]
                if need_to_stop == 1:
                    if frame_cnt_traffic_light < frame_rate * 2: # 2초 동안 느리게 주행 후 정지
                        lane_tracking_slowly(diff)
                        frame_cnt_traffic_light += 1
                    else:
                        car.throttle = 0
                        print("Stop at traffic light")
                else: # Priority3 : other traffic sign
                    if count[1] == 1 and frame_cnt_cross < frame_rate * 3:
                        car.throttle = 0
                        print("Stop")
                        frame_cnt_cross += 1
                    elif count[0] == 1 and count[5] > 0 and frame_cnt_bus < frame_rate * 4: # 버스표지판, 보행자가 같이 인식되면 4초간 일시 정지
                        car.throttle = 0
                        print("Stop for bus passenger")
                        steering_prev = car.steering
                        frame_cnt_bus += 1
                    else: # 무단횡단 보행자도 없고, 표지판도 없고, 신호등도 안보일 때 정상 주행
                        # Initialize cnt for next frame
                        count[0] = 0 # bus_sign
                        count[1] = 0 # crosswalk
                        count[5] = 0 # pedestrian
                        frame_cnt_pedestrian = 0
                        
                        if 초음파센서 거리가 20cm 이내이면:
                            lane_tracking_for_ACC(diff) # 초음파센서의 거리값을 바탕으로 속도를 제어하는 함수, 차간거리 유지에 시간적인 여유가 된다면 앞차량이 멈춰있다가 움직이는 경우에도 가속하면서 추종하도록 기능 추가
                            steering_prev = car.steering
                        else:
                            lane_tracking(diff)
                            steering_prev = car.steering
                        



            # For checking whether object detection is well done
            print("bus count : ",frame_cnt_bus,"cross count : ",frame_cnt_cross," left count : ",frame_cnt_left," right count : ",frame_cnt_right)

            # For tuning lane tracking parameters
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