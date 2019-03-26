import cv2
import numpy as np

img = cv2.imread("/home/saujanya/OCR/practice/final/different processes/0.jpg")
cv2.imshow("Original",img)

img_c=img.copy()
img_c=cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
(t, img_c)=cv2.threshold(img_c,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
cv2.imshow("thresh",img_c)
kernel1 = np.ones((1,1), np.uint8) 
temp_img1 = cv2.morphologyEx(img_c,cv2.MORPH_CLOSE,kernel1,iterations=3)
letter_img1 = cv2.erode(temp_img1,kernel1,iterations=1)
cv2.imshow("Letter image 1",letter_img1)
kernel2 = np.ones((1,1),np.uint8)
temp_img2 = cv2.morphologyEx(img_c,cv2.MORPH_CLOSE,kernel2,iterations=3)
letter_img2 = cv2.erode(temp_img1,kernel2,iterations=1)
cv2.imshow("Letter image 2",letter_img2)
final = np.bitwise_or(letter_img1,letter_img2)
cv2.imshow("Final 1",final)
(_,contours,_)=cv2.findContours(final.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h=cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

cv2.imshow("Final",img)

cv2.waitKey(0)
cv2.destroyAllWindows()