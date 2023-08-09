import cv2 as cv
import numpy as np
import glob

# cap = cv.VideoCapture(0)  # left camera  создает объект, с которого будет происходить захват видео

# ret, frame = cap.read()  # cap.read() в ret возвращает значение, типа Boolean (T/F).
# Если frame прочитан корректно: ret = True.
# cv.imwrite('cam.png', frame)

# cap.release()

# Определение размеров шахматной доски
CHECKERBOARD = (6, 9)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Создание вектора для хранения векторов трехмерных точек для каждого изображения шахматной доски
objpoints = []
# Создание вектора для хранения векторов 2D точек для каждого изображения шахматной доски
imgpoints = []

# Определение мировых координат для 3D точек
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Извлечение пути отдельного изображения, хранящегося в данном каталоге
images = glob.glob('./capture/right/*.jpg')
for fname in images:
    img = cv.imread(fname)
    # h, w, _ = img.shape
    # h_new = math.floor(h / 3.6)
    # w_new = math.floor(w / 3.6)
    # img = cv.resize(img, (w_new, h_new))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Найти углы шахматной доски
    # Если на изображении найдено нужное количество углов, тогда ret = true
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK +
                                            cv.CALIB_CB_NORMALIZE_IMAGE)

    """
    Если желаемый номер угла обнаружен, уточняем координаты
    пикселей и отображаем их на изображениях шахматной доски
    """
    if ret == True:
        objpoints.append(objp)

        # Уточнение координат пикселей для заданных 2d точек
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Нарисовать и отобразить углы
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        cv.imshow('Image', img)
        cv.waitKey(0)

    cv.destroyAllWindows()

# h, w = img.shape[:2]

"""
Выполнение калибровки камеры с помощью передачи значения известных трехмерных точек (объектов)
и соответствующих пиксельных координат обнаруженных углов (imgpoints)
"""

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Ret: \n")
print(ret)
print("-" * 70)
print("Camera matrix: \n")
print(mtx)
print("-" * 70)
print("distCoeffs: \n")
print(dist)
print("-" * 70)
print("Rotation vector: \n")
print(rvecs)
print("-" * 70)
print("Translation vector: \n")
print(tvecs)

# fileCamMtx = open('leftcameraMatrix.txt', 'w', )  # left camera
fileCamMtx = open('rightcameraMatrix.txt', 'w', )  # right camera
fileCamMtx.write(str(mtx))

# fileDistCoeffs = open('leftdistCoeffs.txt', 'w', )  # left camera
fileDistCoeffs = open('rightdistCoeffs.txt', 'w', )  # right camera
fileDistCoeffs.write(str(dist))

'''------------------------------------------------------------------------------------------------------------------'''

img = cv.imread('./capture/right/000444.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('calibresult.jpg', dst)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

cv.imwrite('calibresult.jpg', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objpoints))
'''------------------------------------------------------------------------------------------------------------------'''

# LEFT CAMERA
# Ret:
#
# 0.2753437926322073
# ----------------------------------------------------------------------
# Camera matrix:
#
# [[805.9431052    0.         395.11344783]
#  [  0.         795.45709464 138.23505312]
#  [  0.           0.           1.        ]]
# ----------------------------------------------------------------------
# distCoeffs:
#
# [[-0.05306627 -0.07529091  0.01229103 -0.00106944  0.19452425]]
# ----------------------------------------------------------------------
# Rotation vector:
#
# [array([[-0.02749692],
#        [ 0.03776717],
#        [ 1.56674791]]), array([[-0.04210178],
#        [ 0.02778192],
#        [ 0.72953396]]), array([[0.01054683],
#        [0.07453869],
#        [2.39094484]]), array([[0.02203654],
#        [0.07801377],
#        [2.16331166]]), array([[-0.01765932],
#        [ 0.06083873],
#        [ 2.26438857]]), array([[0.0377212 ],
#        [0.09887835],
#        [3.13437878]]), array([[-0.04592949],
#        [ 0.03272401],
#        [ 1.58523759]]), array([[-0.02155049],
#        [ 0.02568441],
#        [ 1.58758322]]), array([[0.00190332],
#        [0.05146569],
#        [1.56330229]]), array([[-0.02543884],
#        [ 0.06516262],
#        [ 1.58434266]]), array([[-0.02790707],
#        [ 0.03758781],
#        [ 1.56970496]]), array([[-0.00622253],
#        [ 0.03564323],
#        [ 1.58108347]]), array([[0.00209876],
#        [0.06765146],
#        [1.55717486]])]
# ----------------------------------------------------------------------
# Translation vector:
#
# [array([[ 1.51433931],
#        [ 1.49082053],
#        [27.01346967]]), array([[-1.51765832],
#        [-0.96445211],
#        [27.14979967]]), array([[ 2.0883897 ],
#        [ 5.24780656],
#        [26.72056804]]), array([[-3.28267277],
#        [ 0.7823037 ],
#        [27.05527839]]), array([[ 6.41831116],
#        [ 6.08310983],
#        [26.77371737]]), array([[-0.69798028],
#        [ 8.10654792],
#        [26.62636496]]), array([[ 6.78639239],
#        [-3.01512803],
#        [27.28835958]]), array([[ 6.77634143],
#        [ 5.18201836],
#        [26.80423926]]), array([[-4.06637639],
#        [ 5.3687708 ],
#        [26.96840189]]), array([[ 1.47803226],
#        [-3.49479804],
#        [27.24333215]]), array([[ 6.56425936],
#        [ 1.55176529],
#        [26.96762558]]), array([[ 1.6256169 ],
#        [ 5.16265397],
#        [26.78135125]]), array([[-4.35026167],
#        [ 1.19629372],
#        [27.05665604]])]
# total error:  0.03665569313997836


