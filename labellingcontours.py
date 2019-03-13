import numpy as np
import cv2
import pytesseract

THRESHOLD = 200

im=cv2.imread("/home/saujanya/OCR/practice/final/page1.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('imgray', imgray)
img=im.copy()
#blur=cv2.GaussianBlur(imgray,(5,5),0)
#cv2.imshow('blur', blur)

ret, thresh = cv2.threshold(imgray, THRESHOLD, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)
thresh_inv=cv2.bitwise_not(thresh)
cv2.imshow('thresh_inv', thresh_inv)

#for line
kernel = np.ones((1,3), np.uint8)
kernel2 = np.ones((1,4), np.uint8)
temp_img = cv2.morphologyEx(thresh_inv,cv2.MORPH_CLOSE,kernel2,iterations=20)
word_img = cv2.dilate(temp_img,kernel,iterations=40)
(_, contours,_) = cv2.findContours(word_img .copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print ("Number of contours detected = %d" % len(contours))

'''
#for paragraph
kernel = np.ones((3,3), np.uint8)
par_img = cv2.dilate(thresh_inv,kernel,iterations=40)
(_,contours, _) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
'''

cv2.drawContours(im.copy(), contours, -1, (1, 70, 255), 2)
contours=contours[::-1]
i=0
for cnt in contours:
    x,y,w,h=cv2.boundingRect(cnt)
    cv2.rectangle(img,(x-1,y-5),(x+w,y+h),(0,255,0),1)
    cv2.putText(img, str(i),(x-15,y+12),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.3,(0,0,0),1,cv2.LINE_AA)
    i=i+1
    
extract=word_img[y:y+h , x:x+w]
cv2.imshow("Extracted image",extract)
text=pytesseract.image_to_string(extract)
print(text)
cv2.imshow("Contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()