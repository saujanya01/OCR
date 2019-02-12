import cv2

img=cv2.imread("/home/saujanya/OCR/practice/final/noise_removal/line_removed.png",cv2.CV_8UC3)
cv2.imshow("Original",img)

#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

thresh=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,4)
cv2.imshow("Thresholded",thresh)

#dst = cv2.fastNlMeansDenoisingColored(thresh,None,10,10,7,21)
#cv2.imshow("Final",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()