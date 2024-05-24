from __future__ import annotations
import os
import pygame
import cv2
from lib.jetracer.nvidia_racecar import NvidiaRacecar
from IPython.display import display, Image
import threading
from lib.jetcam.utils import bgr8_to_jpeg
from pathlib import Path
from typing import Sequence
import argparse
import datetime
import glob
import logging
import numpy as np
import time
import torch


car = NvidiaRacecar()

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"


# joystick setting
pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

running = True

logging.getLogger().setLevel(logging.INFO)


def draw_boxes(image, pred, classes, colors):
    """Visualize YOLOv8 detection results"""
    for r in pred:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = round(float(box.conf[0]), 2)
            label = int(box.cls[0])

            color = colors[label].tolist()
            cls_name = classes[label]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{cls_name} {score}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 1280,
        height: int = 720,
        _width: int = 640,
        _height: int = 360,
        frame_rate: int = 30,
        flip_method: int = 0,
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = False,
        save: bool = False,
        log: bool = True,
    ) -> None:
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self._width = _width
        self._height = _height
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log
        self.model = None

        # Check if OpenCV is built with GStreamer support
        # print(cv2.getBuildInformation())

        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence) and len(sensor_id) > 1:
            raise NotImplementedError("Multiple cameras are not supported yet")

        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        self.cap = [cv2.VideoCapture(self.gstreamer_pipeline(sensor_id = id), \
        				cv2.CAP_GSTREAMER) for id in self.sensor_id]

        # Make record directory
        if save:
            assert save_path is not None, "Please provide a save path"
            os.makedirs(self.save_path, exist_ok=True) # if path does not exist, create it
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)

            logging.info(f"Save directory: {self.save_path}")

    def gstreamer_pipeline(self, sensor_id: int) -> str:
        """
        Return a GStreamer pipeline for capturing from the CSI camera
        """
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                self.width,
                self.height,
                self.frame_rate,
                self.flip_method,
                self._width,
                self._height,
            )
        )

    def set_model(self, model: YOLO, classes: dict) -> None:
        """
        Set a YOLO model
        """
        self.model = model
        self.classes = classes                
        self.colors = np.random.randn(len(self.classes), 3)
        self.colors = (self.colors * 255.0).astype(np.uint8)
        self.visualize_pred_fn = lambda img, pred: draw_boxes(img, pred, self.classes, self.colors)

    def run(self) -> None:
        """
        Streaming camera feed
        """
        if self.stream:
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            try:
                running = True
                recording = False
                history = False
                
                index = 0
                current_index = 0
                j = 0

                while running:
                    t0 = time.time()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    _, frame = self.cap[0].read()

                    if self.model is not None:
                        # run model
                        pred = self.model(frame)
                        print(pred)
                        
                        # visualize prediction
                        frame_for_yolo = frame
                        self.visualize_pred_fn(frame_for_yolo, pred)

                    if (self.save) and recording:
                        # j = j+1
                        # cv2.imwrite(str(self.save_path / "../{:06d}.format(j)/{:05d}.jpg".format(index)), frame)
                        
                        cv2.imwrite(str(self.save_path / "{:05d}.jpg".format(index)), frame)

                    if self.log:
                        print(f"FPS: {1 / (time.time() - t0):.2f}")

                    if self.stream:
                        cv2.imshow(self.window_title, frame)
                        # file_path = './rec.avi'
                        # fps = self.cap[0].get(cv2.CAP_PROP_FPS)
                        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')            # 인코딩 포맷
                        # width = self.cap[0].get(cv2.CAP_PROP_FRAME_WIDTH)
                        # height = self.cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT)
                        # size = (int(width), int(height))                   # 프레임 크기
                        
                        # out = cv2.VideoWriter(file_path, fourcc, fps, size)
                        # out.write()

                        if cv2.waitKey(1) == ord('q'):
                            break
                        
                    pygame.event.pump()
                    throttle_range = (-0.4, 0.4)
                    steering_range = (-0.4, 0.4)
                    car.throttle_gain = 0.3
                    # motion
                    throttle = -joystick.get_axis(1)
                    throttle = max(throttle_range[0], min(throttle_range[1], throttle))

                    steering = joystick.get_axis(2)
                    steering = max(steering_range[0], min(steering_range[1], steering))

                    car.throttle = throttle
                    car.steering = steering        

                    if joystick.get_button(6):
                        print("image saving start")
                        recording = True
                        if history == True:
                            index = current_index
                        else:
                            index = 0
                        time.sleep(1)
                    if joystick.get_button(7):
                        print("image saving end")
                        recording = False
                        current_index = index
                        history = True
                        time.sleep(1)
                    if joystick.get_button(11):
                        self.cap[0].release()# start button
                        running = False
                    index += 1
            except Exception as e:
                print(e)
            finally:
                self.cap[0].release()
                cv2.destroyAllWindows()

    @property
    def frame(self) -> np.ndarray:
        """
        !!! Important: This method is not efficient for real-time rendering !!!

        [Example Usage]
        ...
        frame = cam.frame # Get the current frame from camera
        cv2.imshow('Camera', frame)
        ...

        """
        if self.cap[0].isOpened():
            return self.cap[0].read()[1]
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--sensor_id',
        type = int,
        default = 0,
        help = 'Camera ID')
    args.add_argument('--window_title',
        type = str,
        default = 'Camera',
        help = 'OpenCV window title')
    args.add_argument('--save_path',
        type = str,
        default = 'record',
        help = 'Image save path')
    args.add_argument('--save',
        action = 'store_true',
        help = 'Save frames to save_path')
    args.add_argument('--stream',
        action = 'store_true',
        help = 'Launch OpenCV window and show livestream')
    args.add_argument('--log',
        action = 'store_true',
        help = 'Print current FPS')
    args.add_argument('--yolo_model_file',
        type = str,
        default = None,
        choices = [None, 'yolov8n.pytorch.pt', 'yolov8n.pytorch.engine'],
        help = 'YOLO model')
    
    args.add_argument('--yolo_pt_file',
        type = str,
        default = None,
        choices = [None, 'yolov8n.pytorch.pt', 'best.pt','best_yolo.pt'],
        help = 'YOLO model')
    
    args = args.parse_args()

    cam = Camera(
        sensor_id = args.sensor_id,
        window_title = args.window_title,
        save_path = args.save_path,
        save = args.save,
        stream = args.stream,
        log = args.log)

    if args.yolo_model_file is not None:
        from ultralytics import YOLO
        classes = YOLO('best.pt', task='detect').names
        model = YOLO('best.pt', task='detect')
        # model = YOLO(args.yolo_model_file, task='detect')
        cam.set_model(model, classes)

    cam.run()
