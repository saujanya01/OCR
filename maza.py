import cv2
import pytesseract
img=cv2.imread("/home/saujanya/Desktop/sss.png")
text=pytesseract.image_to_string(img)
print(text)
cv2.namedWindow('Original',cv2.WINDOW_NORMAL)
cv2.imshow("Original",img)
cv2.waitKey(0)
cv2.destroyAllWindows()