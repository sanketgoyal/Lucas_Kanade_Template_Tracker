import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os

images = []
cv_img = []
path = '/home/sanket/Desktop/ENPM673/Untitled Folder/car/'

for image in os.listdir(path):
    images.append(image)
images.sort()

for image in images:
    img = cv2.imread("%s%s" % (path, image))
    cv_img.append(img)

def warpInv(p):
    inverse_output = np.matrix([[0.1]] * 6)
    val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
    inverse_output[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    inverse_output[1, 0] = (-p[1, 0]) / val
    inverse_output[2, 0] = (-p[2, 0]) / val
    inverse_output[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    inverse_output[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
    inverse_output[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val
    return inverse_output

def LucasKanade(in_temp, in_temp_a, rectangle, s=np.zeros(2)):
    
    x1, y1, x2, y2 = rectangle[0], rectangle[1], rectangle[2], rectangle[3]
    temp_y, temp_x = np.gradient(in_temp_a)
    ds = 1
    thresh = 0.001

    while np.square(ds).sum() > thresh:
        s_x, s_y = s[0], s[1]
        w_x1, w_y1, w_x2, w_y2 = x1 + s_x, y1 + s_y, x2 + s_x, y2 + s_y

        u = np.linspace(x1, x2, 87)
        v = np.linspace(y1, y2, 36)
        u0, v0 = np.meshgrid(u, v)

        w_u = np.linspace(w_x1, w_x2, 87)
        w_v = np.linspace(w_y1, w_y2, 36)
        w_u0, w_v0 = np.meshgrid(w_u, w_v)

        x = np.arange(0, in_temp.shape[0], 1)
        y = np.arange(0, in_temp.shape[1], 1)

        spline = RectBivariateSpline(x, y, in_temp)
        S = spline.ev(v0, u0)

        spline_a = RectBivariateSpline(x, y, in_temp_a)
        img_warp = spline_a.ev(w_v0, w_u0)


        error = S - img_warp
        img_error = error.reshape(-1, 1)


        spline_x = RectBivariateSpline(x, y, temp_x)
        warpTemp_x = spline_x.ev(w_v0, w_u0)

        spline_y = RectBivariateSpline(x, y, temp_y)
        warpTemp_y = spline_y.ev(w_v0, w_u0)
        temp = np.vstack((warpTemp_x.ravel(), warpTemp_y.ravel())).T


        jac_matrix = np.array([[1, 0], [0, 1]])


        hess_matrix = temp @ jac_matrix

        H = hess_matrix.T @ hess_matrix



        ds = np.linalg.inv(H) @ (hess_matrix.T) @ img_error


        s[0] += ds[0, 0]
        s[1] += ds[1, 0]

    stop = s
    return stop

rectangle = [118,100, 338,280]
b = rectangle[3] - rectangle[1]
l = rectangle[2] - rectangle[0]
rectangle0 = copy.deepcopy(rectangle)
capture_in = cv_img[0]
capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
capture_gray_in = cv2.equalizeHist(capture_gray_in)
#out = cv2.VideoWriter('Car.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (720, 480))

for i in range(0, len(cv_img)-1):

    index = i

    if index == 100:

        rectangle = [145, 113, 345, 275]

        l = rectangle[2] - rectangle[0]
        b = rectangle[3] - rectangle[1]
        
        rectangle0 = copy.deepcopy(rectangle)
        capture_in = cv_img[100]
        capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
        capture_gray_in = cv2.equalizeHist(capture_gray_in)

    if index == 150:
        rectangle = [200,113,395,275]

        l = rectangle[2] - rectangle[0]
        b = rectangle[3] - rectangle[1]
        
        rectangle0 = copy.deepcopy(rectangle)
        capture_in = cv_img[150]
        capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
        capture_gray_in = cv2.equalizeHist(capture_gray_in)

    if index == 200:

        rectangle = [295,125,446,240]

        l = rectangle[2] - rectangle[0]
        b = rectangle[3] - rectangle[1]
        
        rectangle0 = copy.deepcopy(rectangle)
        capture_in = cv_img[200]
        capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
        capture_gray_in = cv2.equalizeHist(capture_gray_in)

    capture = cv_img[index]
    maincapture = capture
    capture_gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    capture_gray = cv2.equalizeHist(capture_gray)
    cv2.rectangle(maincapture,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[0])+l,int(rectangle[1])+b),(255,0,0),3)
    cv2.imshow('Tracking Car', capture)
    #out.write(capture)

    capture_next = cv_img[index+1]
    capture_gray_next = cv2.cvtColor(capture_next, cv2.COLOR_BGR2GRAY)
    capture_gray_next = cv2.equalizeHist(capture_gray_next)

    in_temp_x = capture_gray_in / 255.
    in_temp = capture_gray / 255.
    in_temp_a = capture_gray_next / 255.
    stop = LucasKanade(in_temp_x, in_temp_a, rectangle0)
    rectangle[0] = stop[0] + rectangle0[0]
    rectangle[1] = stop[1] + rectangle0[1]
    rectangle[2] = stop[0] + rectangle0[2]
    rectangle[3] = stop[1] + rectangle0[3]

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
#out.release()







