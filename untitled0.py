import cv2
import numpy as np


img=cv2.imread('/home/saujanya/OCR/practice/final/page1.png')
cv2.imshow("Original",img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=np.bitwise_not(gray)
#cv2.imshow("grayscale",gray)
b_fil=cv2.bilateralFilter(gray,7,50,50)
#g1_blur=cv2.GaussianBlur(gray,(7,7),0)
#cv2.imshow("Bilateral Filter",b_fil)
#cv2.imshow("G1 Blur",g1_blur)
#thresh_gray=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
#thresh_gray1=cv2.adaptiveThreshold(g1_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
(T,thresh)=cv2.threshold(b_fil,200,255,cv2.THRESH_BINARY)
#thresh_b_fil=cv2.adaptiveThreshold(b_fil,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,4)
#cv2.imshow("Gray Thtresh",thresh_gray)
#cv2.imshow("Gray Thtresh g1_blur",thresh_gray1)
#cv2.imshow(" Thresholded",thresh)

#kernel=np.zeros((3,3),np.uint8)
#open_img=cv2.morphologyEx(thresh_b_fil,cv2.MORPH_CLOSE,kernel)
#cv2.imshow("Morph open",open_img)



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
(t,finalrot)=cv2.threshold(rotated,200,255,cv2.THRESH_BINARY)
#cv2.imwrite("Straight.png",final)

#text=pytesseract.image_to_string(final)
#print(text)

cv2.imshow('Rotated',finalrot)
cv2.imshow('Original',img)
b_fil=cv2.bilateralFilter(finalrot,7,50,50)
(T,thresh1)=cv2.threshold(b_fil,200,255,cv2.THRESH_BINARY_INV)
horizontal_img=thresh1
vertical_img=thresh1
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(50,1))
horizontal_img=cv2.erode(horizontal_img,kernel,1)
horizontal_img=cv2.dilate(horizontal_img,kernel,1)

kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,50))
vertical_img=cv2.erode(vertical_img,kernel,1)
vertical_img=cv2.dilate(vertical_img,kernel,1)

mask_img=horizontal_img+vertical_img;
kernel_mask_h=cv2.getStructuringElement(cv2.MORPH_RECT,(1,20))
kernel_mask_v=cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))
final_mask_v=cv2.dilate(vertical_img,kernel_mask_v,1)
final_mask_h=cv2.dilate(horizontal_img,kernel_mask_h,1)
mask_img=final_mask_h+final_mask_v
not_i=np.bitwise_not(mask_img)
final=np.bitwise_and(not_i,thresh1)

final1=cv2.bilateralFilter(final,15,50,50)

#Writing image
cv2.imwrite("line_removed.png",np.bitwise_not(final1))


#cv2.imshow("Mask",mask_img)
#cv2.imshow("Hor",horizontal_img)
#cv2.imshow("Ver",vertical_img)
#cv2.imshow("Final",np.bitwise_not(final))
cv2.imshow("Final1",np.bitwise_not(final1))
#text=pytesseract.image_to_string(final1)
#print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()