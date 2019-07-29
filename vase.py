import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os

images = []
cv_img = []
path = '/home/sanket/Desktop/ENPM673/Untitled Folder/vase/'

for image in os.listdir(path):
    images.append(image)
images.sort()

for image in images:
    img = cv2.imread("%s%s" % (path, image))
    cv_img.append(img)

    
def LucasKanade(in_temp, in_temp_a, rectangle, s=np.zeros(2)):

    x1, y1, x2, y2, x3, y3, x4, y4 = rectangle[0], rectangle[1], rectangle[2], rectangle[3], rectangle[4], rectangle[5], rectangle[6], rectangle[7]
    temp_y, temp_x = np.gradient(in_temp_a)
    ds = 1
    thresh = 0.0001

    while np.square(ds).sum() > thresh:

        s_x, s_y = s[0], s[1]
        w_x1, w_y1, w_x2, w_y2, w_x3, w_y3, w_x4, w_y4 = x1 + s_x, y1 + s_y, x2 + s_x, y2 + s_y, x3 + s_x, y3 + s_y, x4 + s_x, y4 + s_y

        u1 = np.linspace(x1, x3, 87)
        v1 = np.linspace(y1, y3, 36)
        u2 = np.linspace(x4, x2, 87)
        v2 = np.linspace(y2, y4, 36)
        u = np.union1d(u1, u2)
        v = np.union1d(v1, v2)
        u0, v0 = np.meshgrid(u, v)

        w_u1 = np.linspace(w_x1, w_x3, 87)
        w_v1 = np.linspace(w_y1, w_y3, 36)
        w_u2 = np.linspace(w_x4, w_x2, 87)
        w_v2 = np.linspace(w_y2, w_y4, 36)
        w_u = np.union1d(w_u1, w_u2)
        w_v = np.union1d(w_v1, w_v2)
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

    stp = s
    return stp

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

rectangle1 = [124,91,172,91,172,150,124,150]
rectangle2 = [62, 48, 85, 48, 85, 74, 62, 74]
rectangle3 = [31, 22, 43, 22, 43, 37, 31, 37]
rectangle4 = [16, 13, 21, 13, 21, 18, 16, 18]
rectangle10 = copy.deepcopy(rectangle1)
rectangle20 = copy.deepcopy(rectangle2)
rectangle30 = copy.deepcopy(rectangle3)
rectangle40 = copy.deepcopy(rectangle4)

capture_in = cv_img[0]
capture_in = cv2.GaussianBlur(capture_in, (9,9), 0)
capture_gray_in_1 = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
capture_gray_in_2 = cv2.pyrDown(capture_gray_in_1)
capture_gray_in_3 = cv2.pyrDown(capture_gray_in_2)

for i in range(0, len(cv_img)-1):
    index = i
    capture = cv_img[index]
    capture_gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)


    cv2.line(capture, (int(rectangle1[0]), int(rectangle1[1])), (int(rectangle1[2]), int(rectangle1[3])), (255,0,0), 3)
    cv2.line(capture, (int(rectangle1[0]), int(rectangle1[1])), (int(rectangle1[6]), int(rectangle1[7])), (255, 0, 0), 3)
    cv2.line(capture, (int(rectangle1[2]), int(rectangle1[3])), (int(rectangle1[4]), int(rectangle1[5])), (255, 0, 0), 3)
    cv2.line(capture, (int(rectangle1[6]), int(rectangle1[7])), (int(rectangle1[4]), int(rectangle1[5])), (255, 0, 0), 3)

    cv2.imshow('Tracking_Vase', capture)
    capture_next = cv_img[index+1]
    capture_next = cv2.GaussianBlur(capture_next, (9,9), 0)
    capture_gray_next1 = cv2.cvtColor(capture_next, cv2.COLOR_BGR2GRAY)
    capture_gray_next2 = cv2.pyrDown(capture_gray_next1)
    capture_gray_next3 = cv2.pyrDown(capture_gray_next2)
    
    in_temp_x = capture_gray_in_3 / 255.

    in_temp_a = capture_gray_next3 / 255.
    stop1 = LucasKanade(in_temp_x, in_temp_a, rectangle30)
    
    in_temp_x = capture_gray_in_2 / 255.

    in_temp_a = capture_gray_next2 / 255.
    stop2 = LucasKanade(in_temp_x, in_temp_a, rectangle20, s = np.array(stop1)*2)
    
    in_temp_x = capture_gray_in_1 / 255.

    in_temp_a = capture_gray_next1 / 255.
    stop = LucasKanade(in_temp_x, in_temp_a, rectangle10, s = np.array(stop1)*4 + np.array(stop2)*2)
    
    rectangle1[0] = stop[0] + rectangle10[0]
    rectangle1[1] = stop[1] + rectangle10[1] 
    
    rectangle1[2] = stop[0] + rectangle10[2]
    rectangle1[3] = stop[1] + rectangle10[3]
    
    rectangle1[4] = stop[0] + rectangle10[4]
    rectangle1[5] = stop[1] + rectangle10[5]
    
    rectangle1[6] = stop[0] + rectangle10[6]
    rectangle1[7] = stop[1] + rectangle10[7] 
    
    if (index > 16 and index < 30) or (index > 56 and index < 68) or (index > 81 and index < 99):
        rectangle1[1] = 90
        rectangle1[3] = 90
        
    if index > 16 and index< 30 or (index > 56 and index < 68) or (index > 81 and index < 99):
        rectangle1[0] = rectangle1[0] - 20
        rectangle1[6] = rectangle1[6] - 20
        
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
