# Программа для настройки цветового фильтра
import cv2 as cv
import numpy as np
import pyzed.sl as sl


def nothing(*arg):
    pass


cv.namedWindow("result")  # создаем главное окно
cv.namedWindow("settings")  # создаем окно настроек

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv.createTrackbar('hue_1', 'settings', 0, 255, nothing)
cv.createTrackbar('satur_1', 'settings', 0, 255, nothing)
cv.createTrackbar('value_1', 'settings', 0, 255, nothing)
cv.createTrackbar('hue_2', 'settings', 255, 255, nothing)
cv.createTrackbar('satur_2', 'settings', 255, 255, nothing)
cv.createTrackbar('value_2', 'settings', 255, 255, nothing)

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Prepare new image size to retrieve half-resolution images
image_size = zed.get_camera_information().camera_resolution
image_size.width = image_size.width / 2
image_size.height = image_size.height / 2

image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

while True:
    if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        image_ocv = image_zed.get_data()

        hsv = cv.cvtColor(image_ocv, cv.COLOR_BGR2HSV)

        # считываем значения бегунков
        h1 = cv.getTrackbarPos('hue_1', 'settings')
        s1 = cv.getTrackbarPos('satur_1', 'settings')
        v1 = cv.getTrackbarPos('value_1', 'settings')
        h2 = cv.getTrackbarPos('hue_2', 'settings')
        s2 = cv.getTrackbarPos('satur_2', 'settings')
        v2 = cv.getTrackbarPos('value_2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        thresh = cv.inRange(hsv, h_min, h_max)

        cv.imshow("result", thresh)

        if cv.waitKey(1) == ord('q'):
            break

cv.destroyAllWindows()
zed.close()

