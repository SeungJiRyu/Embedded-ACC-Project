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

######################################################## README ########################################################
# 0(bus) : 이전 프레임과 달리 인식되지 않으면 sign control 안에서 초기화
# 1(crosswalk) : 이전 프레임과 달리 인식되지 않으면 sign control 안에서 초기화
# 2(green) : red/yellow light가 인식되면 sign control 안에서 초기화
# 3(left) : 직선모델이 사용되면 메인 루프 안에서 초기화
# 4(pedestrian) : 이전 프레임과 달리 인식되지 않으면 sign control 안에서 초기화
# 5(red) : green light가 인식되면 sign control 안에서 초기화
# 6(right) : 직선모델이 사용되면 메인 루프 안에서 초기화
# 7(yellow) : green light가 인식되면 sign control 안에서 초기화
# >>> 객체 인식하지 않는 frame이더라도 초기화되지 않고 가장 최근에 detection한 값을 유지
#
# 2024.06.13 테스트 전에 해야할 일
# 1. 이미지 프로세싱 완료 안될 시 : 보행자 인지 시 급정거 코드에서 boundary1,2값 임의 설정 필요
# 2. 필요시 PID gain tuning, box_area 조절, 적절한 속도 조절
# 3. dictionary로 cnt되는 값 보기 좋게 정리
# 4. 차선 상에서 보행자랑 버스정류장은 confidence score값이 낮으면, sign_control에서 count하는 조건값 수정 필요
########################################################################################################################

os.environ["SDL_VIDEODRIVER"] = "dummy"

# Constant for GPIO
TRIG = 16
ECHO = 22

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
            label = f"{classes[int(box.cls[0])]} {round(float(box.conf[0]), 2)}, AREA : {(x2-x1)*(y2-y1)}"
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
car.throttle_gain = 0.32 # 0.34 # reset 0.315
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
    global throttle_ACC
    car.steering = -np.sign(diff) * abs((diff / 140)) ** 1.3 * steering_range[1] # 140
    if abs(car.steering - steering_prev) > 0.2:
        throttle_ACC -= abs(car.steering - steering_prev) * 0.005
        
    car.throttle = max(throttle_range[0], min(throttle_range[1], throttle_ACC))

# ACC Controller
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
            return None
        time.sleep(0.00001)

    start_time = time.time()
    while GPIO.input(ECHO) == GPIO.HIGH:
        stop = time.time()
        if stop - start_time > 0.02:  # 20ms 후 타임아웃
            print("Timeout waiting for ECHO to go LOW")
            return None
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

Kp = 0.003  # 비례 이득
Ki = 0  # 적분 이득
Kd = 0  # 미분 이득

previous_error = 0
integral = 0

target_distance = 25  # 목표 거리 (cm)
emergency_distance = 5  # 급정거 거리 (cm)

def pid_control(current_distance, dt):
    global previous_error, integral

    error = -(target_distance - current_distance)
    integral += error * dt
    derivative = (error - previous_error) / dt
    previous_error = error

    output = Kp * error + Ki * integral + Kd * derivative
    return output

# Variable for initializing bus, crosswalk, pedestrian
prev_bus = 0
prev_crosswalk = 0
prev_pedestrian = 0

# 0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pesdestrian', 5: 'red', 6: 'right', 7:'yellow'
# size : bus_sign / crosswalk, left, right/ red, yellow, green / pedestrian
def sign_control(pred_sign, box, confidence):
    global x1_pes, x2_pes, frame_cnt_traffic_light, whether_stop, prev_bus, prev_crosswalk, prev_pedestrian
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    box_area = w * h
    # Save temporary prev cnt for bus, crosswalk, pedestrian
    prev_bus = count[0]
    prev_crosswalk = count[1]
    prev_pedestrian = count[4]
    
    # Detection for bus_sign
    if box_area > 1000 and confidence > 0.4:
        if pred_sign == 'bus_sign':
            count[0] += 1
    # Detection for traffic sign(expect bus)
    if box_area > 3000 and confidence > 0.5: #### 3000 or 4500
        if pred_sign == 'crosswalk':
            count[1] += 1
        elif pred_sign == 'left' and count[3] == 0:
            count[3] += 1
        elif pred_sign == 'right' and count[6] == 0:
            count[6] += 1

    # Detection for pedestrian
    if box_area > 1000  and confidence > 0.4: # 보행자 인지시 급정거
        if pred_sign == 'pedestrian':
            count[4] += 1
        if box_area > 10000:
            x1_pes = x1
            x2_pes = x2
            whether_stop += 1
            
    # Detection for traffic light
    if box_area > 1500 and confidence > 0.5: ############### 신호등을 인식하고 교차로 앞에서 멈추는 적절한 박스 사이즈 결정 필요
        if pred_sign == 'red' and count[5] == 0:
            count[2] = 0
            count[5] += 1
        elif pred_sign == 'yellow' and count[7] == 0:
            count[2] = 0
            count[7] += 1
        elif pred_sign == 'green' and count[2] == 0:
            count[2] += 1
            count[5] = 0
            count[7] = 0

    # Initialize bus, crosswakl, pedestrian
    if prev_pedestrian == count[4]:
        count[4] = 0
        whether_stop = 0
    if prev_crosswalk == count[1]:
        count[1] = 0
    if prev_bus == count[0]:
        count[0] = 0



# Initialize flag values 0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pesdestrian', 5: 'red', 6: 'right', 7:'yellow'
count = [0, 0, 0, 0, 0, 0, 0, 0]
whether_stop = 0

running = True

# Model paths
yolo_pth = "./pt/YOLO_ver9.pt"
alexnet_pth = "./pt/version4_alexnet_2.pth"
alexnet_left_pth = "./pt/merged_left_alexnet_epoch30.pth"
alexnet_right_pth = "./pt/merged_right_alexnet_epoch30.pth"
sign_dict = {0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pedestrian', 5: 'red', 6: 'right', 7:'yellow'}
count_dict = {'bus_sign' : 0, 'crosswalk' : 0, 'green' : 0, 'left' : 0, 'pedestrian' : 0, 'red' : 0, 'right' : 0, 'yellow' : 0}
# count_dict['cross_walk'] += 1

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
frame_rate = 15 ##########################################


frame_cnt_bus = 0
frame_cnt_bus2 = 0
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


###################################################### Main ########################################################

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


if cap.isOpened():
    stream = False
    index = 1
    try:
        previous_time = time.time()
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
                    draw_boxes(frame_for_cap, pred_YOLO[0], class_for_yolo, colors_for_yolo)
                    # cv2.imshow("Output", frame_for_cap)
                    # cv2.waitKey(1)
                    
                    for i in range(num_objects):
                        class_id = classes[i]
                        # Count for each object
                        pred_sign = sign_dict[class_id]
                        confidence = confidences[i]
                        box = bounding_boxes[i]
                        sign_control(pred_sign, box, confidence)
            
# flag count values -  0: 'bus_sign', 1: 'crosswalk', 2: 'green', 3: 'left', 4: 'pesdestrian', 5: 'red', 6: 'right', 7:'yellow'
            
            # [Priority] no collision with pedestrian > no breaking the traffic law(traffic light) >  ...
            # Priority1 : check whether pedestrain is detected
            # 바운딩 박스가 차선 안에 들어와 있으면(= 충돌이 예상되면) 정지, 일단은 3초동안 일시 정지하도록 구현함 (일시 정지 시간을 없애봐도 괜찮을 듯)
            if whether_stop > 0 and ((boundary1 < x1_pes < boundary2) or (boundary1 < x2_pes < boundary2)) and frame_cnt_pedestrian < frame_rate * 3:
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
                if count[3] == 1 and frame_cnt_left < frame_rate * time_for_left_right:
                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_left_model, frame_width, frame_height)
                    print("Left on")
                    #### Red or Green light detect order ####
                    if not (count[5] or count[7]):
                        frame_cnt_left += 1
                elif count[6] == 1 and frame_cnt_right < frame_rate * time_for_left_right:
                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_right_model, frame_width, frame_height)
                    print("Right on")
                    if not (count[5] or count[7]):
                        frame_cnt_right += 1

                else:
                    # Initialize count for left, right sign
                    count[3] = 0
                    count[6] = 0
                    # Initialize frame count left, right
                    if frame_cnt_left >= frame_rate * time_for_left_right:
                        frame_cnt_left = 0
                    if frame_cnt_right >= frame_rate * time_for_left_right:
                        frame_cnt_right = 0

                    out_of_range, intersection, xpre, ypre = output_model_ALEXNET(frame, ALEXNET_model, frame_width, frame_height)

                # Calculate difference for controller
                diff = frame_width // 2 - xpre
                
                # Priority2 : Check for need to stop(red or yellow light detection)
                need_to_stop = count[5] or count[7]
                if need_to_stop == 1:
                    if frame_cnt_traffic_light < frame_rate * 2.3: ########################### 2.3초 동안 느리게 주행 후 정지 : 몇초간 느리게 할지 논의 필요
                        lane_tracking_slowly(diff)
                        frame_cnt_traffic_light += 1
                    else :
                        car.throttle = 0  
                        print("Stop at traffic light")
                
                        
                else: # Priority3 : green light and other traffic sign
                    frame_cnt_traffic_light = 0

                    if count[1] > 0 and frame_cnt_cross < frame_rate * 3: # crosswalk
                        car.throttle = 0
                        print("Stop")
                        frame_cnt_cross += 1
                    elif count[0] > 0 and count[4] > 0 and frame_cnt_bus < frame_rate * 4.5: # bus_sign, pedestrian 동시에 인식되면 3초간 일시 정지 ######## 몇초간 멈추게 할지 논의 필요
                        if frame_cnt_bus2 < frame_rate * 1.5:
                            lane_tracking_slowly(diff)
                            frame_cnt_bus2 += 1
                            frame_cnt_bus += 1
                        else: ############# 만약 버스정류장 2개 놓게되면 frame_cnt_bus, bus2 초기화하는 코드 추가 필요 #############3
                            car.throttle = 0
                            print("Stop for bus passenger")
                            steering_prev = car.steering
                            frame_cnt_bus += 1

                    else: # 무단횡단 보행자도 없고, 표지판도 없고, 신호등도 안보일 때 정상 주행
                        result_distance = distance_check()
                        
                        if result_distance is not None:
                            filtered_result = apply_ema_filter(result_distance)
                            # print("distance = {:.2f} cm, filtered distance = {:.2f} cm".format(result_distance, filtered_result))

                            current_time = time.time()
                            dt = current_time - previous_time
                            previous_time = current_time

                            if filtered_result < emergency_distance: # 무조건 급정거
                                car.throttle = 0
                                print("Emergency stop")
                            elif filtered_result > target_distance: #선행차량이 목표거리 안에 x, 목표 속도 주행
                                lane_tracking(diff)
                                steering_prev = car.steering
                            else : # 선행차량이 목표거리 안에 존재할 때, 속도 제어
                                # PID 제어를 통한 속도 조정
                                throttle_adjustment = pid_control(filtered_result, dt)
                                throttle_ACC = max(throttle_range[0], min(throttle_range[1], throttle + throttle_adjustment))
                                print(f"throttle_ACC = {throttle_ACC}")
                                lane_tracking_for_ACC(diff)
                                steering_prev = car.steering
                                print(f"Adjusting speed: throttle = {car.throttle}")
                            
               
                        

            # For checking whether object detection is well done
            print("bus count : ",frame_cnt_bus,"cross count : ",frame_cnt_cross," left count : ",frame_cnt_left," right count : ",frame_cnt_right, " traffic count : ",frame_cnt_traffic_light)
            print(count)

            # For tuning lane tracking parameters
            # print(diff)
            draw_circles(frame_for_cap, xpre, ypre)
            draw_boxes(frame_for_cap, pred_YOLO, class_for_yolo, colors_for_yolo)
            cv2.imshow("YOLO", frame_for_cap)
            cv2.waitKey(1)
            

            if stream:
                cv2.imwrite(f"./images/trial_1/{index:05d}.jpg", frame_for_cap)
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