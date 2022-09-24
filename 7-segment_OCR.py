from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(0, 1, 1, 1, 0, 1, 1): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 0 ,0, 1, 0): 7,
	(1, 0, 1, 0, 0, 1, 1): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

# 이미지 불러오고 원하는 영역 선택 후 영역 자르기
def load_image(path):

	global width, height
	original_img = cv2.imread(path)
	x_pos, y_pos, width, height = cv2.selectROI("DRAG RECTANGULAR",original_img,False)
	cutted_img = original_img[y_pos:y_pos+height, x_pos:x_pos+width]

	# 흑백으로 바꾸고 Blur(흐림) 적용
	gray_img = cv2.cvtColor(cutted_img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

	return blurred, gray_img, cutted_img


# 이미지 이진화
def preprocessing(img):

	ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 이진화의 threshold 값은 OTSU 알고리즘 적용
	k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
	dst = cv2.dilate(thr, k) # 숫자 부분을 두껍게 만들어 인식률 높이기

	return dst


# 숫자 위치 찾기
def find_digits_positions(img, originalimg):

	global NUMOFDIGITS
	NUMOFDIGITS = 0
	#윤곽선 검출
	cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	digitCnts = []

	# 일정 수준의 크기와 비율에 해당하는 윤곽선을 배열에 저장
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		if (w<= width) and (h >= height*(0.5) and h <= height*(1.2)):
			if (1<= h/w <= 10):
				digitCnts.append(c)
	
	for contour in digitCnts:
		cv2.drawContours(originalimg, [contour], 0, (255, 0, 0), 3)

	return digitCnts

# 숫자 인식
def recognize_digit(digit_positions, output_img, input_img, original_img):
	digitCnts = contours.sort_contours(digit_positions, method = "left-to-right")[0]
	digits = []
	for c in digitCnts:
		(x, y, w, h) = cv2.boundingRect(c)

		# 윤곽선의 세로/가로 비율이 클 경우 숫자 1로 인식
		if 2.5 < h/w <= 10:
			roi = input_img[y:y+h, x:x+w]
			digits.append(1)
			cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
			cv2.putText(original_img, str(1), (x + 10, y + 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
			continue

		# 1이 아닐 경우 segment를 나누어 숫자 인식
		roi = input_img[y:y+h, x:x+w]
		(roiH, roiW) = roi.shape
		(dW, dH) = (int(roiW * 0.3333), int(roiH * 0.2))

		segments = [
				((dW, 0), (dW*2, dH)),	# top
				((0, dH), (dW, dH*2)),	# top-left
				((dW*2, dH), (w, dH*2)),	# top-right
				((dW, dH*2 ) , (dW*2, dH*3)), # center 
				((0, dH*3), (dW, dH*4)),	# bottom-left 
				((dW*2, dH*3), (w, dH*4)),	# bottom-right
				((dW, dH*4), (dW*2, h))	# bottom
			]
		on = [0] * len(segments)

		for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
			# 관심영역을 추출하고 임계값을 넘기는 픽셀의 수 세기
			segROI = roi[yA:yB, xA:xB]
			total = cv2.countNonZero(segROI)
			area = (xB - xA) * (yB - yA)

			# 만약 임계값을 넘기는 픽셀이 전체 픽셀의 40% 이상이면 'ON' 상태
			if total / float(area) > 0.4:
				on[i]= 1

		digit = DIGITS_LOOKUP[tuple(on)] 
		digits.append(digit)
		cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.putText(original_img, str(digit), (x + 10, y + 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	
	cv2.imshow('OUTPUT',original_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return digits

def main():
	for i in range(1, 16):
		path = ("segment_ex/segment-%s.jpg" %i)
		original = cv2.imread(path)
		blurred, gray_img, cutted_img = load_image(path)
		dst = preprocessing(blurred)
		digits_positions = find_digits_positions(dst, cutted_img)
		digits= recognize_digit(digits_positions, blurred, dst, cutted_img)
		print(digits)

if __name__ == '__main__':
	main()