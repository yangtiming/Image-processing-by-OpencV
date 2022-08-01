import numpy as np
import os
import cv2
import time
import math
import pyzbar.pyzbar as pyzbar
from PIL import Image,ImageDraw,ImageFont
def change_cv2_draw(image,txtss,x,y):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("simsun.ttc",15,encoding="unic")
    draw.text((x, y -15), txtss, (255,0, 0), font=font)
    img222 = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    return img222,font
def decodeDisplay(image,pic1):

    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(pic1, (x, y), (x + w, y + h), (0, 0, 255), 2)

        barcodeData = barcode.data.decode("utf-8 ")
        barcodeType = barcode.type

        cv2.imshow('imgcod2e', pic1)
        text = "{} ({})".format(barcodeData, barcodeType)
        pic1,text = change_cv2_draw(pic1, text,x,y)
        print(text)
        #cv2.putText(pic1, text, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,
         #          .5, (0, 0, 125), 2)

        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        cv2.imshow('imgcode',pic1)



def CalcEuclideanDistance(point1, point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance


def CalcFourthPoint(point1, point2, point3):
    D = (point1[0] + point2[0] - point3[0], point1[1] + point2[1] - point3[1])
    return D


def JudgeBeveling(point1, point2, point3):
    try:

        dist1 = CalcEuclideanDistance(point1, point2)
        dist2 = CalcEuclideanDistance(point1, point3)
        dist3 = CalcEuclideanDistance(point2, point3)
        dist = [dist1, dist2, dist3]
        max_dist = dist.index(max(dist))
        if max_dist == 0:
            D = CalcFourthPoint(point1, point2, point3)
        elif max_dist == 1:
            D = CalcFourthPoint(point1, point3, point2)
        else:
            D = CalcFourthPoint(point2, point3, point1)
        return D
    except:
        return


def jietu1(c, image):
    flag = 0
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
        t = max(hight, width)
        cropImg = image[y1 - 20:y1 + t + 20, x1 - 20:x1 + t + 20]
        cv2.imshow("000", cropImg)

    except:
        return


# ====================================================================================#
def compute_center(contours, i):
    '''计算轮廓中心点'''
    M = cv2.moments(contours[i])

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


# ====================================================================================#
def compute_1(contours, i, j):
    '''最外面的轮廓和子轮廓的比例'''
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    if area2 == 0:
        return False
    if (area1 > area2) == 1:
        ratio = area1 * 1.0 / area2

    if abs(ratio - 49.0 / 25) < 2:
        return 1
    return False


def compute_2(contours, i, j):
    '''子轮廓和子子轮廓的比例'''

    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])

    if area2 == 0:
        return False
    if area1 > area2:
        ratio = area1 * 1.0 / area2
    if abs(ratio - 25.0 / 9) < 2:
        return True
    return False


# ====================================================================================#
def drawcontour(frame0, frame, hierarchy, contours):
    list_x = []
    list1 = []
    list2 = []
    list3 = []
    flag = 0
    for i in range(1, len(contours) - 2):
        child = hierarchy[0][i][2]
        child_child = hierarchy[0][child][2]
        if child != -1 and hierarchy[0][child][2] != -1:

            hierarchyremember = child
            if compute_1(contours, hierarchyremember - 1, hierarchyremember):
                if compute_2(contours, hierarchyremember, hierarchyremember + 1):
                    cv2.drawContours(frame, contours, hierarchyremember - 1, (0, 255, 0), 2)
                    cv2.drawContours(frame, contours, hierarchyremember, (0, 0, 255), 2)
                    cv2.drawContours(frame, contours, hierarchyremember + 1, (0, 0, 255), 2)
                    x1, y1 = compute_center(contours, hierarchyremember + 1)
                    x2, y2 = compute_center(contours, hierarchyremember - 1)
                    x3, y3 = compute_center(contours, hierarchyremember)
                    list1.append([x2, y2])
                    list2.append((x2, y2))
                    flag = 1
    try:
        a = JudgeBeveling(list2[0], list2[1], list2[2])
        list_x = list_x + list1 + [a]
    except:
        pass

    c = np.array(list_x)
    jietu1(c, frame)

    return flag


# ====================================================================================#

def threshold(pic):
    gra = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    gray1 = gra.copy()
    pic1=pic.copy()
    im = decodeDisplay(gray1,pic1)
    # binary88=cv2.blur(gra,(3,3))
    cv2.imshow("blur", gra)
    ret, binary = cv2.threshold(gra, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold", binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return hierarchy, contours


# ====================================================================================#


def video_start():
    print("执行开始".center(25, "-"))
    cap = cv2.VideoCapture("C:\\Users\\Administrator\\Desktop\\139.mp4")  # 获取视频地址
    # /Users/timingyang/Desktop/pythonshiyan/share2/cloud_classification_all/139.MP4
    totalFrameNumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(totalFrameNumber)
    start = time.perf_counter()
    regnize_total = 0
    total = 0

    while cap.isOpened():
        print('ok')
        dur = time.perf_counter() - start
        ret, frame = cap.read()
        ret1, frame0 = cap.read()
        if ret == False:
            break
        # ------------------------------------------#
        hierarchy, contours = threshold(frame)
        flag = drawcontour(frame0, frame, hierarchy, contours)
        total = total + 1

        if flag == 1:
            regnize_total = regnize_total + 1
            print("识别百分比[{:2}/{:2}]".format( regnize_total, total))

        cv2.imshow('img', frame)
        # cv2.waitKey(0)
        # ------------------------------------------#
        c = cv2.waitKey(10)
        time.sleep(0.1)
        if c & 0xFF == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()
    print("——————————————执行结束	——————————————")


video_start()