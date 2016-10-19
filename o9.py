#!/usr/bin/env python    
# encoding: utf-8  
## API Docs:http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html?highlight=canny  

import cv2    
import numpy as np   
from matplotlib import pyplot as plt
import math


BLUE = (255,0,0)

# 已经有两个点，求一元一次方程y=kx+d的系数k和常数项d，判断地平线
def find_skyline(x1, y1, x2, y2, img):
    k = 1.0 * (y2 - y1) / (x2 - x1)
    d = 1.0 * (y1*x2 - y2*x1) / (x2-x1)   
    return (k, d)

## 去除暗角
## 形成一个圆形，圆形外部开始逐渐变白，最后和原始图形叠加
## types=0是叠加一个圆形，=1是叠加一个矩形。可能是方法不当，=1时效果不好
def vignette_filter(img, pixels_falloff = 0, types=0):
    height, width = img.shape
    radius = max(width, height) / 2.0 * 0.95
    radius = min(width, height) / 2.0 * 0.95
    # pixels_falloff = 0.1
    row_ctr = height / 2;
    col_ctr = width / 2
    max_img_rad = math.sqrt(row_ctr * row_ctr + col_ctr * col_ctr)
    res = img.copy()
    
    if types:
        trow = pixels_falloff
        lcol = pixels_falloff
        brow = img.shape[0] - pixels_falloff * 2
        rcol = img.shape[1] - pixels_falloff * 2
    for i in range(height):
        for j in range(width):
            dh = abs(i - row_ctr)
            dw = abs(j - col_ctr)            
            if not types:
                dis = math.sqrt(dh * dh + dw * dw)
                if dis > radius:
                    if dis > radius + pixels_falloff:                   
                        res[i, j] = img[i, j] * (dis) / radius
                    else:
                        sigma = (dis - radius) / pixels_falloff
                        res[i, j] = img[i, j] * (1 - sigma * sigma)
                else:               
                    pass
            else:
                dis1 = min(abs(i - trow), abs(i - brow))
                dis2 = min(abs(j - lcol), abs(j-rcol))
                if i<= brow and i >= trow and j >= lcol and j <= rcol:
                    pass
                else:
                    sigma = (dis1 + dis2) * (dis1 + dis2) / (dis1 * dis1 + dis2 * dis2)
                    res[i, j] = img[i, j] * sigma
    return res


def affline_rotate(img, pts1, pts2):
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

if __name__ == "__main__":  
    
    img = cv2.imread("1.png", 0)  #Canny只能处理灰度图，所以将读取的图像转成灰度图  

    cv2.imshow('original image', img)
    img2 = vignette_filter(img, 0.3)   
    img = cv2.blur(img, (5, 5))    
    img2 = cv2.blur(img2, (10, 10))    
    img = cv2.addWeighted(img, 0.80, img2, 0.20, 1)    
    # cv2.imshow('after vignette_filter', img)

    ### pre-process and use candy
    img = cv2.blur(img, (15, 15))
    clahe = cv2.createCLAHE(clipLimit=2.00, tileGridSize=(11, 11))
    img = clahe.apply(img)   
    ret2,detected_edges = cv2.threshold(img, 10, 230, cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    edges = cv2.Canny(detected_edges,0.1,1.0, apertureSize = 3)  
    dst = cv2.bitwise_and(img,img,mask = edges)

    ## 因为地平线为一条直线，该部分可以发现直线，并将地平线以上部分置为背景色
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,10,np.pi/180,100,minLineLength,maxLineGap)
    a1, b1, a2, b2 = (0, 0, 0, 0)
    dis = 0
    ## 采用最长的直线作为地平线。可能会有很多短直线
    for x1,y1,x2,y2 in lines[0]:
        if (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) > dis:
             a1, b1, a2, b2 = (x1,y1,x2,y2)  
    (k, d)= find_skyline(a1,b1,a2,b2, img)
    print "line: ", k,d
    # 将地平线以上置为空
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            pos = int(k * j + d)
            if i <= pos + 4:
                dst[i, j] = 0

    cv2.imshow('canny lines', dst)
    # cv2.imwrite('houghlines5.jpg',img)

    ## 根据canny line，将原图形中边界标出，
    original = cv2.imread("1.png")
    for row in range(original.shape[0]):	
    	for bt in range( original.shape[1]):
    		if dst[row, bt] == 0:
    			pass	
    		else:    					
    			original[row, bt] = BLUE
    ## 结果已经在original里

    ## 根据k,d,算出原始的放射点
    srcy1 = img.shape[1]
    srcx1 = k * srcy1 + d
    srcx2 = img.shape[0]
    srcy2 = 0
    srcx3 = img.shape[0]
    srcy3 = img.shape[1]
    pts1 = np.float32([[int(srcy1) , int(srcx1)],[srcy2,srcx2],[srcy3,srcx3]])  
    pts2 = np.float32([[img.shape[1] * 0.9, 0],[0, img.shape[0] / 7],[img.shape[1]*0.75, img.shape[0]]])

    img = cv2.imread('1.png', 0) 
    img = affline_rotate(img,pts1, pts2)
    detected_edges = affline_rotate(detected_edges, pts1, pts2) 
    result = affline_rotate(original, pts1,pts2) 	
    cv2.imshow('affine original image', img)	
    cv2.imshow('affine edges', detected_edges)
    cv2.imshow('affine canny result', result)
    # cv2.imwrite('result.png', original)

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()