import numpy as np
import os
import cv2  
import time 
import math


def waiji(poin_a,poin_b,poin_c):
	
	veca = np.array(poin_a)
	vecb = np.array(poin_b)
	vecc = np.array(poin_c)
	vec_ab = veca - vecb
	vec_ac = veca - vecc
	result = vec_ac[0]*vec_ab[1]-vec_ac[1]*vec_ab[0]
	print('waiji_result:',result)
	if result < 0:
		
		return poin_a,poin_b,poin_c
	else:
		return poin_a,poin_c,poin_b
		

def touyingbianhuan(poind_a,poind_b,poind_c,poind_d,image):
	
	img=image.shape
	vec0 = np.array(poind_a)
	vecx = np.array(poind_b)
	vecy = np.array(poind_c)
	distance1 = np.linalg.norm(vec0 - vecx)
	distance2 = np.linalg.norm(vec0 - vecy)
	dmax=int(max(distance1,distance2))
	
	#srct=[[100,200],[100,300],[200,200],[200,300]]
	print("qrcode_four_point:",poind_a,poind_a,poind_c,poind_d)
	#src = np.array([[poind_a[0]-15,poind_a[1]-15],[poind_b[0]+15,poind_b[1]-15],[poind_d[0]+15,poind_d[1]+15],[poind_c[0]-15,poind_c[1]+15]], np.float32)
	src = np.array([poind_a,poind_b,poind_d,poind_c], np.float32)
	
	
	dst = np.array([[0,0],[dmax,0],[dmax,dmax],[0,dmax]],dtype="float32")
	A1 = cv2.getPerspectiveTransform(src, dst)
	
	
	d1 = cv2.warpPerspective(image, A1, (dmax,dmax), borderValue = 255)
	
	cv2.imshow("xuanzhuan_image",d1)
def CalcEuclideanDistance(point1,point2):
	vec1 = np.array(point1)
	vec2 = np.array(point2)
	distance = np.linalg.norm(vec1 - vec2)
	
	return distance
def CalcFourthPoint(point1,point2,point3): #pint3为A点
	D = [point1[0]+point2[0]-point3[0],point1[1]+point2[1]-point3[1]]
	return D
def JudgeBeveling(point1,point2,point3):
	try:
		
		dist1 = CalcEuclideanDistance(point1,point2)
		dist2 = CalcEuclideanDistance(point1,point3)
		dist3 = CalcEuclideanDistance(point2,point3)
		dist = [dist1, dist2, dist3]
		max_dist = dist.index(max(dist))
		if max_dist == 0:
			D = CalcFourthPoint(point1,point2,point3)
			pointa=point3
			pointb=point2
			pointc=point1
		elif max_dist == 1:
			D = CalcFourthPoint(point1,point3,point2)
			pointa=point2
			pointb=point3
			pointc=point1
		else:
			D = CalcFourthPoint(point2,point3,point1)
			pointa=point1
			pointb=point3
			pointc=point2
		return D,pointa,pointb,pointc
	except:
		return 
def jietu1(list_x,image):
	c=np.array(list_x)
	flag=0
	try:
		rect = cv2.minAreaRect(c)
		box = np.int0(cv2.boxPoints(rect))
		Xs = [i[0] for i in box]
		Ys = [i[1] for i in box]
		x1 = min(Xs)
		x2 = max(Xs)
		y1 = min(Ys)
		y2 = max(Ys)
		hight = y2 - y1
		width = x2 - x1
		t=max(hight,width)
		cropImg = image[y1-20:y1+t+20, x1-20:x1+t+20]
		cv2.imshow("jietu_without xuanzhuan", cropImg)
		
	except:
		return
#====================================================================================#
def compute_center(contours,i):
	'''计算轮廓中心点'''
	M=cv2.moments(contours[i])
	
		
	cx = int(M['m10'] / M['m00'])
	cy = int(M['m01'] / M['m00'])
	return cx,cy



#====================================================================================#
def compute_1(contours,i,j):
	'''最外面的轮廓和子轮廓的比例'''
	area1 = cv2.contourArea(contours[i])
	area2 = cv2.contourArea(contours[j])
	if area2==0:
		
		return False
	if (area1>area2)==1:
		ratio = area1 * 1.0 / area2
		
		
	if abs(ratio - 49.0 / 25)<2:
		return 1
	return False
def compute_2(contours,i,j):
	'''子轮廓和子子轮廓的比例'''
	
	area1 = cv2.contourArea(contours[i])
	area2 = cv2.contourArea(contours[j])
	
	if area2==0:
		return False
	if area1>area2:
		ratio = area1 * 1.0 / area2
	if abs(ratio - 25.0 / 9)<2:
		return True
	return False

#====================================================================================#
def drawcontour(frame0,frame,hierarchy,contours):
	list_x=[]
	list1=[]
	list2=[]
	list3=[]
	flag=0
	for i in range(1,len(contours)-2):
		child = hierarchy[0][i][2]
		child_child=hierarchy[0][child][2]
		if child!=-1 and hierarchy[0][child][2]!=-1:
		
			hierarchyremember = child 
			if compute_1(contours,hierarchyremember-1,hierarchyremember):
				if compute_2(contours,hierarchyremember,hierarchyremember+1):
			
					cv2.drawContours(frame,contours,hierarchyremember-1,(0,255,0),2)	
					cv2.drawContours(frame,contours,hierarchyremember,(0,0,255),2)
					cv2.drawContours(frame,contours,hierarchyremember+1,(0,0,255),2)			
					x1,y1=compute_center(contours,hierarchyremember+1)
					x2,y2=compute_center(contours,hierarchyremember-1)
					x3,y3=compute_center(contours,hierarchyremember)
					list1.append([x2,y2])
					list2.append([x2,y2])
					flag=1
	
	try:
		
		D,point_a,point_b,point_c = JudgeBeveling(list2[0],list2[1],list2[2])
		
		poind_a,poind_b,poind_c = waiji(point_a,point_b,point_c)

		touyingbianhuan(poind_a,poind_b,poind_c,D,frame0)
		
		list_x=list_x+list1+[D]	
	except:
		pass
	
	
			
	jietu1(list_x,frame)
	
	return flag
#====================================================================================#

def threshold(pic):
	gra = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)  
	#binary88=cv2.blur(gra,(3,3))
	cv2.imshow("blur",gra)
	ret, binary = cv2.threshold(gra,100,255,cv2.THRESH_BINARY)
	cv2.imshow("threshold",binary)
	contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	return  hierarchy,contours
#====================================================================================#


def video_start():
	print("执行开始".center(25, "-"))
	cap = cv2.VideoCapture('/Users/timingyang/Desktop/pythonshiyan/share2/cloud_classification_all/139.MP4')#获取视频地址
	#/Users/timingyang/Desktop/pythonshiyan/share2/cloud_classification_all/139.MP4
	totalFrameNumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(totalFrameNumber)
	start = time.perf_counter()
	regnize_total=0
	total=0
	while 1:
		dur = time.perf_counter() - start
		ret, frame =cap.read()
		ret1,frame0=cap.read()	
		if ret == False:
			break
#------------------------------------------#
		hierarchy,contours = threshold(frame)
		flag=drawcontour(frame0,frame,hierarchy,contours)
		total=total+1
		if flag == 1:
			regnize_total = regnize_total + 1
			print("识别占帧数百分比[{:2}/{:2}]".format( regnize_total,totalFrameNumber))
			print("识别占循环次数百分比[{:2}/{:2}]".format( regnize_total,total))
			
		cv2.imshow('img',frame)
		#cv2.waitKey(0)
#------------------------------------------#
		c = cv2.waitKey(10)
		if  c & 0xFF == ord('q'):
			break  
	cap.release()
	
	cv2.destroyAllWindows()
	print("——————————————执行结束	——————————————")


video_start()