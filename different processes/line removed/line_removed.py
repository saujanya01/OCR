import cv2
import numpy as np
import pytesseract

img=cv2.imread('/home/saujanya/OCR/practice/final/0.jpg')
(h,w)=img.shape[:2]
width=700
r=h/w
dim=(width,int(width*r))
resized=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
img=resized
cv2.imshow("Original",img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("grayscale",gray)
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
horizontal_img=thresh_b_fil
vertical_img=thresh_b_fil

'''kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(100,1))
horizontal_img=cv2.erode(horizontal_img,kernel,1)
horizontal_img=cv2.dilate(horizontal_img,kernel,1)
'''
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,200))
vertical_img=cv2.erode(vertical_img,kernel,1)
vertical_img=cv2.dilate(vertical_img,kernel,1)

mask_img=vertical_img;

#kernel_mask_h=cv2.getStructuringElement(cv2.MORPH_RECT,(1,40))
kernel_mask_v=cv2.getStructuringElement(cv2.MORPH_RECT,(40,1))
final_mask_v=cv2.dilate(vertical_img,kernel_mask_v,1)
#final_mask_h=cv2.dilate(horizontal_img,kernel_mask_h,1)
mask_img=final_mask_v

not_i=np.bitwise_not(mask_img)
final=np.bitwise_and(not_i,thresh_b_fil)

final1=cv2.bilateralFilter(final,15,50,50)

#Writing image
cv2.imshow("line_removed.png",np.bitwise_not(final1))

final1 = np.bitwise_or(gray,np.bitwise_not(final1))
#cv2.imshow("Mask",mask_img)
#cv2.imshow("Hor",horizontal_img)
#cv2.imshow("Ver",vertical_img)
cv2.imshow("Final",np.bitwise_not(final))

(T,thresh)=cv2.threshold(final1,200,255,cv2.THRESH_BINARY)
'''kernel_fin=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
thresh=cv2.erode(thresh,kernel_fin,1)
kernel_fin1=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
thresh=cv2.dilate(thresh,kernel_fin1,1)'''

cv2.imshow("Final1",thresh)
text=pytesseract.image_to_string(thresh)
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()