import cv2
import numpy as np
import pytesseract

img=cv2.imread('/home/saujanya/OCR/practice/final/different processes/0.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(T,thresh)=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
cv2.imshow('After thresholding',thresh)

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
cv2.imwrite("Straight.png",final)

text=pytesseract.image_to_string(final)
print(text)

cv2.imshow('Rotated',final)
cv2.imshow('Original',img)

cv2.waitKey(0)
cv2.destroyAllWindows()