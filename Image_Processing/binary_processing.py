import cv2
import os
import numpy as np
import datetime

# 이미지가 있는 폴더 경로 지정
folder_path = 'path/to/your/image/folder'
output_folder_path = 'path/to/save/gray/images'

# 폴더 내 파일 리스트 가져오기
file_list = os.listdir(folder_path)

# 각 파일을 순회하면서 흑백처리
for file_name in file_list:
    # 파일의 전체 경로를 생성
    file_path = os.path.join(folder_path, file_name)

    # 이미지 읽기
    image = cv2.imread(file_path)
    
    # 이미지가 제대로 읽혔는지 확인
    if image is None:
        print(f'Error reading {file_path}')
        continue
    
    # 이미지를 흑백으로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cropping image
    src_points = np.array([[0,180],[640,180],[640,360],[0,360]],dtype=np.float32)
    mask = np.zeros((360,640), dtype = np.uint8)
    cv2.fillPoly(mask, [src_points.astype(np.int32)],255)
    cropped_img = cv2.bitwise_and(image,image,mask=mask)
    
    # 이미지를 흑백으로 변환
    gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray_image,(5,5),0)
    
    _,threshold = cv2.threshold(blurred,200,255,cv2.THRESH_BINARY_INV) # WB : 0~255 200기준으로 WB(White,Black)변환
   
   # Denoise
    denoised = cv2.erode(threshold,None,iterations=2)
    denoised = cv2.dilate(denoised,None,iterations=2)
            
    # 흑백 이미지를 저장할 경로 생성
    gray_file_path = os.path.join(output_folder_path, file_name)
    
     # 흑백 이미지 저장
    cv2.imwrite(gray_file_path, denoised)
    
    print(f'Successfully processed {file_name}')

print('All images have been processed.')