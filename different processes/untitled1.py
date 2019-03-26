import cv2
import numpy as np
import pytesseract

img=cv2.imread('/home/saujanya/OCR/practice/final/test_images/2.png')
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
image1 = np.bitwise_not(final1)
image2 = np.bitwise_not(final1)
image3 = np.bitwise_not(final1)

# hardcoded assigning of output images for the 3 input images
output1_letter = image1.copy()
output1_word = image1.copy()
output1_line = image1.copy()
output1_par = image1.copy()
output1_margin = image1.copy()

output2_letter = image2.copy()
output2_word = image2.copy()
output2_line = image2.copy()
output2_par = image2.copy()
output2_margin = image2.copy()


output3_letter = image3.copy()
output3_word = image3.copy()
output3_line = image3.copy()
output3_par = image3.copy()
output3_margin = image3.copy()

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3	= cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# clean the image using otsu method with the inversed binarized image
ret1,th1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret2,th2 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret3,th3 = cv2.threshold(gray3,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#processing letter by letter boxing
def process_letter(thresh,output):	
	# assign the kernel size	
	kernel = np.ones((2,1), np.uint8) # vertical
	# use closing morph operation then erode to narrow the image	
	temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)
	# temp_img = cv2.erode(thresh,kernel,iterations=2)		
	letter_img = cv2.erode(temp_img,kernel,iterations=1)
	
	# find contours 
	(_,contours, _) = cv2.findContours(letter_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	# loop in all the contour areas
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	


#processing letter by letter boxing
def process_word(thresh,output):	
	# assign 2 rectangle kernel size 1 vertical and the other will be horizontal	
	kernel = np.ones((2,1), np.uint8)
	kernel2 = np.ones((1,4), np.uint8)
	# use closing morph operation but fewer iterations than the letter then erode to narrow the image	
	temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)
	#temp_img = cv2.erode(thresh,kernel,iterations=2)	
	word_img = cv2.dilate(temp_img,kernel2,iterations=1)
	
	(_,contours, _) = cv2.findContours(word_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	

#processing line by line boxing
def process_line(thresh,output):	
	# assign a rectangle kernel size	1 vertical and the other will be horizontal
	kernel = np.ones((1,5), np.uint8)
	kernel2 = np.ones((2,4), np.uint8)	
	# use closing morph operation but fewer iterations than the letter then erode to narrow the image	
	temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel2,iterations=2)
	#temp_img = cv2.erode(thresh,kernel,iterations=2)	
	line_img = cv2.dilate(temp_img,kernel,iterations=5)
	
	(_,contours, _) = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	

#processing par by par boxing
def process_par(thresh,output):	
	# assign a rectangle kernel size
	kernel = np.ones((5,5), 'uint8')	
	par_img = cv2.dilate(thresh,kernel,iterations=3)
	
	(_,contours, _) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)

	return output	

#processing margin with paragraph boxing
def process_margin(thresh,output):	
	# assign a rectangle kernel size
	kernel = np.ones((20,5), 'uint8')	
	margin_img = cv2.dilate(thresh,kernel,iterations=5)
	
	(_,contours, _) = cv2.findContours(margin_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)

	return output


# processing and writing the output
output1_letter = process_letter(th1,output1_letter)
output1_word = process_word(th1,output1_word)
output1_line = process_line(th1,output1_line)
# special case for the 5th output because margin with paragraph is just the 4th output with margin
cv2.imshow("output1_letter",output1_letter)
cv2.imshow("output1_word",output1_word)
cv2.imshow("output1_line",output1_line)
cv2.imwrite("output/letter/output1_letter.jpg", output1_letter)	
cv2.imwrite("output/word/output1_word.jpg", output1_word)
cv2.imwrite("output/line/output1_line.jpg", output1_line)
output1_par = process_par(th1,output1_par)
cv2.imwrite("output/par/output1_par.jpg", output1_par)
output1_margin = process_margin(th1,output1_par)
cv2.imwrite("output/margin/output1_margin.jpg", output1_par)

output2_letter = process_letter(th2,output2_letter)
output2_word = process_word(th2,output2_word)
output2_line = process_line(th2,output2_line)

cv2.imwrite("output/letter/output2_letter.jpg", output2_letter)	
cv2.imwrite("output/word/output2_word.jpg", output2_word)
cv2.imwrite("output/line/output2_line.jpg", output2_line)
output2_par = process_par(th2,output2_par)
cv2.imwrite("output/par/output2_par.jpg", output2_par)
output2_margin = process_margin(th2,output2_par)
cv2.imwrite("output/margin/output2_margin.jpg", output2_par)

output3_letter = process_letter(th3,output3_letter)
output3_word = process_word(th3,output3_word)
output3_line = process_line(th3,output3_line)

cv2.imwrite("output/letter/output3_letter.jpg", output3_letter)	
cv2.imwrite("output/word/output3_word.jpg", output3_word)
cv2.imwrite("output/line/output3_line.jpg", output3_line)
output3_par = process_par(th3,output3_par)
cv2.imwrite("output/par/output3_par.jpg", output3_par)
output3_margin = process_margin(th3,output3_par)
cv2.imwrite("output/margin/output3_margin.jpg", output3_par)

cv2.imshow("output letter", output1_letter)
cv2.imshow("output word", output1_word)
cv2.imshow("output line", output1_line)
cv2.imshow("output par", output1_par)
cv2.imshow("output margin", output1_par)

text=pytesseract.image_to_string(final1)
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()