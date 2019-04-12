#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:09:09 2019

@author: saujanya and naman
"""

import pytesseract
from wand.image import Image as wi
import cv2
import numpy as np

from tkinter import Tk, Button, filedialog, Label


root = Tk()
y = ""
root.title("OCR_Project")

def openfiledialog():
    global y
    y = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("pdf","*.pdf"),("all files","*.*")))

def check_path():
    global y
    print (y)

file_button = Button(root, text='Select File', command=openfiledialog)
file_button.pack()

exit_button = Button(root, text='Let it do the magic :pppp', command=root.destroy)
exit_button.pack()
w = Label(root, text="\nSelect the file path\n\n")
w.pack()

# Button(root, text="Print current saved path", command = check_path).pack()


root.mainloop()
path = y
print(path)
#User enters path of image
# path = str(input("Enter path of the PDF : \n"))
st=""
#conversion of PDF to image in jpeg format
pdf = wi(filename=path,resolution=200)
pdfImage=pdf.convert('jpeg')
n=0
for img in pdfImage.sequence:
    page=wi(image=img)
    page.save(filename=str(n)+".jpg")
    n=n+1

index = path.rfind('/')
path1 = path[:index+1]
for i in range(n):
    image_path = path1+str(i)+'.jpg'
    img=cv2.imread(image_path)
    
    #resizing of image
    
    (h,w)=img.shape[:2]
    if w>700:
        width=700
        r=h/w
        dim=(width,int(width*r))
        resized=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
        img=resized
    #cv2.imwrite(img,"/home/saujanya/OCR/practice/final/resizing/resize"+str(i)+".jpg")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    original = gray
    #Skew Correction
    
    (T,thresh)=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('After thresholding',thresh)
    
    cord=np.column_stack(np.where(thresh>0))
    angle=cv2.minAreaRect(cord)[-1]
    if angle < -45:
        angle=-(angle+90)
    else:
        angle=-angle
    
    (h,w)=img.shape[:2]
    center=(w//2 , h//2)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    rotated=cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    (t,final)=cv2.threshold(rotated,200,255,cv2.THRESH_BINARY)
    #cv2.imshow('rotated',final)
    img = final
    #cv2.imwrite(img,"/home/saujanya/OCR/practice/final/skew correction/skew_corr"+str(i)+".jpg")
    #(t,final)=cv2.threshold(rotated,200,255,cv2.THRESH_BINARY)
    
    #Noise Removal
    
    #gray_r=cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    
    #final image after noise removal is img_nr
    
    #layout analysis and line removal
    THRESHOLD = 200
    #cv2.imshow("Original",img)
    b_fil=cv2.bilateralFilter(gray,7,50,50)
    #g1_blur=cv2.GaussianBlur(gray,(7,7),0)
    #cv2.imshow("Bilateral Filter",b_fil)
    #cv2.imshow("G1 Blur",g1_blur)
    #thresh_gray=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
    #thresh_gray1=cv2.adaptiveThreshold(g1_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
    thresh_b_fil=cv2.adaptiveThreshold(b_fil,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
    #cv2.imshow("Gray Thtresh",thresh_gray)
    #cv2.imshow("Gray Thtresh g1_blur",thresh_gray1)
    #cv2.imshow("Bilateral Filter Thresholded",thresh_b_fil)
    '''
    kernel=np.zeros((3,3),np.uint8)
    open_img=cv2.morphologyEx(thresh_b_fil,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("Morph open",open_img)
    '''
    cv2.imwrite("/home/saujanya/OCR/practice/final/with_line.jpg",np.bitwise_not(thresh_b_fil))
    #Part of line removal
    horizontal_img=thresh_b_fil
    vertical_img=thresh_b_fil

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(100,3))
    horizontal_img=cv2.erode(horizontal_img,kernel,1)
    horizontal_img=cv2.dilate(horizontal_img,kernel,1)

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,100))
    vertical_img=cv2.erode(vertical_img,kernel,1)
    vertical_img=cv2.dilate(vertical_img,kernel,1)

    mask_img=vertical_img+horizontal_img

    kernel_mask_h=cv2.getStructuringElement(cv2.MORPH_RECT,(1,100))
    kernel_mask_v=cv2.getStructuringElement(cv2.MORPH_RECT,(100,1))
    final_mask_v=cv2.dilate(vertical_img,kernel_mask_v,1)
    final_mask_h=cv2.dilate(horizontal_img,kernel_mask_h,1)
    mask_img=final_mask_v+final_mask_h
    '''
    cv2.imshow("Vertical image",vertical_img)
    cv2.imshow("Horizontal image",horizontal_img)
    cv2.imshow("Mask image",mask_img)
    '''
    not_i=np.bitwise_not(mask_img)
    '''
    cv2.imshow("not_i",not_i)
    cv2.imshow("thresh b fil",thresh_b_fil)
    '''
    final=np.bitwise_and(not_i,thresh_b_fil)
    #cv2.imshow("Line removed",final)

    final1=cv2.bilateralFilter(final,15,50,50)

    #Writing image
    #cv2.imshow("line_removed.png",np.bitwise_not(final1))

    final1 = np.bitwise_or(gray,np.bitwise_not(final1))
    #cv2.imshow("Mask",mask_img)
    #cv2.imshow("Hor",horizontal_img)
    #cv2.imshow("Ver",vertical_img)
    #cv2.imshow("Final",np.bitwise_not(final))

    (T,thresh)=cv2.threshold(final1,200,255,cv2.THRESH_BINARY)
    '''kernel_fin=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    thresh=cv2.erode(thresh,kernel_fin,1)
    kernel_fin1=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    thresh=cv2.dilate(thresh,kernel_fin1,1)'''

    #cv2.imshow("Final1",thresh)
    img = thresh
    img=img.copy()
    #blur=cv2.GaussianBlur(imgray,(5,5),0)
    #cv2.imshow('blur', blur)

    #imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgray', imgray)
    ret, thresh = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)
    #cv2.imshow('thresh', thresh)
    thresh_inv=cv2.bitwise_not(thresh)
    #cv2.imshow('thresh_inv', thresh_inv)
    cv2.imwrite("/home/saujanya/OCR/practice/final/line_removal.jpg",thresh)

    #for line
    kernel = np.ones((1,3), np.uint8)
    kernel2 = np.ones((1,4), np.uint8)
    temp_img = cv2.morphologyEx(thresh_inv,cv2.MORPH_CLOSE,kernel2,iterations=50)
    line_img = cv2.dilate(temp_img,kernel,iterations=60)
    (_, contours,_) = cv2.findContours(line_img .copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print ("Number of contours detected = %d" % len(contours))

    '''
    #for paragraph
    kernel = np.ones((3,3), np.uint8)
    par_img = cv2.dilate(thresh_inv,kernel,iterations=40)
    (_,contours, _) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    '''

    cv2.drawContours(img.copy(), contours, -1, (1, 70, 255), 2)
    contours=contours[::-1]
    i=0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000:
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(original,(x-1,y-5),(x+w,y+h),(0,255,0),1)
            cv2.putText(original, str(i),(x-15,y+12),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.3,(0,0,0),1,cv2.LINE_AA)
            cv2.imwrite("/home/saujanya/OCR/practice/final/layout.jpg",original)
            i=i+1
            extract=original[y:y+h , x:x+w]
            text=pytesseract.image_to_string(extract)
            st=st+text+"\n"
    
    st=st+"\n"+"\n"+"-------------------------------------"+"\n"+"\n"
    #cv2.imshow("Contours", img)
    
f=open('/home/saujanya/Desktop/emma.txt','w')
f.write(st)
f.close()
print(st)