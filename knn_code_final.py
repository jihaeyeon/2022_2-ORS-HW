# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:49:06 2022

@author: HaeYeonJI
"""

import numpy as np
import os.path
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from scipy.spatial import distance_matrix
import pandas as pd

# x,y 좌표 -> excel data
# 0. data import ##############################################################
fname_img = "jeju_data.bmp"
os.path.isfile(fname_img)

plt.figure(dpi=1000)
img = rasterio.open(fname_img)                # 영상 불러오기
show(img)                                     # 영상 띄우기
#print(type(img))                             # 영상 data type 확인
I = img.read()                                # 영상 화소값들을 array 형태로 배열
#print(I.shape)                               # 영상 화소값의 dimension 확인  (k,j,i)
#print(type(I))                               # data type 확인
I = np.swapaxes(I, 0, 2)      #image matrix   # i,j,k 로 재 배열             (i, j, k)
X = I.reshape((600*650, 3))   #data matrix    # (i*j, k)


# 1. data select ##############################################################

# 1-1. xy coordinate - 100 data -----------------------------------------------
fname_excel = './data_xy.xlsx'

df = pd.read_excel(fname_excel, header=0)
dff = np.array(df)

xy_flower = dff[:,4].astype(int)
y_flower = np.ones([100,1]) # y=1 : 유채꽃
y_flower = np.where(y_flower==1, 'flower', y_flower).astype(str) # y=1 은 flower 로 바꾸기

xy_green = dff[:,7].astype(int)
y_green = 2*np.ones([100,1]) # y=2 : 녹지
y_green = np.where(y_green==2, 'green', y_green).astype(str)    # y=2 은 green 로 바꾸기

xy_house = dff[:,10].astype(int)
y_house = 3*np.ones([100,1]) # y=3 : 건물
y_house = np.where(y_house==3, 'house', y_house).astype(str)     # y=3 은 house 로 바꾸기

xy_road = dff[:,13].astype(int)
y_road = 4*np.ones([100,1]) # y=4 : 길 
y_road = np.where(y_road==4, 'road', y_road).astype(str)         # y=3 은 road 로 바꾸기

# 추가 25 data select ---------------------------------------------------------

xy2_flower = dff[:25,18].astype(int)
y2_flower = np.ones([25,1]).astype(int)
y2_flower = np.where(y2_flower==1, 'flower', y2_flower).astype(str)

xy2_green = dff[:25,21].astype(int)
y2_green = 2*np.ones([25,1]).astype(int)
y2_green = np.where(y2_green==2, 'green', y2_green).astype(str) 

xy2_house = dff[:25,24].astype(int)
y2_house = 3*np.ones([25,1]).astype(int)
y2_house = np.where(y2_house==3, 'house', y2_house).astype(str)

xy2_road = dff[:25,27].astype(int)
y2_road = 4*np.ones([25,1]).astype(int)
y2_road = np.where(y2_road==4, 'road', y2_road).astype(str) 

# 1-2. RGB(image) coordinate

X_RGB_flower = X[xy_flower]  # 유채꽃
X_RGB_green = X[xy_green]    # 녹지
X_RGB_house = X[xy_house]    # 건물
X_RGB_road = X[xy_road]      # 길 

# 추가 25 data select ---------------------------------------------------------
X2_RGB_flower = X[xy2_flower]
X2_RGB_green = X[xy2_green].astype(int)
X2_RGB_house = X[xy2_house].astype(int)
X2_RGB_road = X[xy2_road].astype(int)


# 1-3. XX행렬=X2 , YY행렬=Y2, RGB 행렬 = rgb2
y2 = np.concatenate((y_flower, y_green, y_house, y_road, y2_flower, y2_green, y2_house, y2_road), axis=0)                   
rgb2 = np.concatenate((X_RGB_flower, X_RGB_green, X_RGB_house, X_RGB_road, X2_RGB_flower, X2_RGB_green, X2_RGB_house, X2_RGB_road),axis=0).astype(int)


# 1-5.Distance matrix #########################################################

D=distance_matrix(rgb2, rgb2)                          # select 한 point의 모든 rbg 에 대해 distance 계산 및 matrix 생성
ind_sort = np.argsort(D)                               # 각 행 마다 D의 값이 작은 순서 대로 해당 index 를 출력 

                                                                                #-----------------------------------
                                                                                # 예                               ㅣ
                                                                                # index        :  0  1   2   3     ㅣ
                                                                                # value        : 10 15  19  12     ㅣ
                                                                                # sort output  :  0, 3, 1, 2       ㅣ
                                                                                #-----------------------------------

ind_nearest= ind_sort[:,1]                           #[:, k] sort 분류 후 가장 value 가 낮은(거리가 가까운)=1열 (0열은 본인, 즉 value=0 이므로)
                         # k=1인 경우 ind_sort[:,1]    # k=3인 경우 ind_sort[:,1:4] 가장 가까운 3개 까지 나열  
# 1-6.Result #########################################################
result = y2[ind_nearest]


# 학습데이터 확인 ---------------------------------------------------------------

result_flower = result[0:100,:]   # Index 0 ~ 99
result_green = result[100:200,:]  # Index 100 ~ 199
result_house = result[200:300,:]  # Index 200 ~ 299
result_road = result[300:400,:]   # Index 300 ~ 399

# 검증데이터 확인 ---------------------------------------------------------------
knn_result_flower = result[400:425,:] # Index 400 ~ 424
knn_result_green = result[425:450,:]  # Index 425 ~ 449
knn_result_house = result[450:475,:]  # Index 450 ~ 474
knn_result_road = result[475:500,:]   # Index 475 ~ 499