import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import math

help_string = "[s] Save side by side image, [d] Save Depth, [n] Change Depth format, " \
              "[p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

# Define Tag
# id_to_find = 100
marker_size = 5  # см


def point_cloud_format_name():
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing")


def depth_format_name():
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing")


def save_point_cloud(zed, filename):
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved:
        print("Done")
    else:
        print("Failed... Please check that you have permissions to write on disk")


def save_depth(zed, filename):
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved:
        print("Done")
    else:
        print("Failed... Please check that you have permissions to write on disk")


def save_sbs_image(zed, filename):
    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)


def process_key_event(zed, key):
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68:
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == 110 or key == 78:
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
    elif key == 109 or key == 77:
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1
    else:
        a = 0


def print_help():
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")
    print(" Press 'q' to quit")


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    Id = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(Id - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# For two markers. Inverse perspective
def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)
    # print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


# Get the camera calibration path
calib_path = ''
camera_matrix = np.loadtxt(calib_path + 'ZEDleftcameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'ZEDleftdistCoeffs.txt', delimiter=',')

# 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# Define the aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()  # Marker detection parameters

# Font for the text in the image
font = cv2.FONT_HERSHEY_COMPLEX


def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2:
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    # init.depth_mode = sl.DEPTH_MODE.QUALITY
    # init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.CENTIMETER
    init.depth_minimum_distance = 30
    init.depth_maximum_distance = 400

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width / 2
    image_size.height = image_size.height / 2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    key = ' '
    while key != 113:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image, depth measure in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # Displaying depth image
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)

            # Get and print distance value in cm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(image_zed.get_width() / 2)
            y = round(image_zed.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)

            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])

            err, depth_value = depth_image_zed.get_value(x, y)

            # Coordinates x and y
            i = 395
            j = 175

            # # Get the 3D point cloud values for pixel (i, j)
            err, point3D = point_cloud.get_value(i, j)
            x1 = point3D[0]
            y1 = point3D[1]
            z1 = point3D[2]
            color = point3D[3]
            distance1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            depth = depth_image_zed.get_data()
            point_cloud_ocv = point_cloud.get_data()

            # 4ch to 3ch - RGBA --> RGB
            image = cv2.cvtColor(image_ocv, cv2.COLOR_RGBA2RGB)
            # depth_image_ocv = cv2.cvtColor(depth_image_rgba, cv2.COLOR_RGBA2RGB)

            # Convert in gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # remember, OpenCV stores color images in Blue, Green, Red

            # HSV color
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            lb = np.array([0, 140, 90])  # lower and upper bound thresholds
            ub = np.array([255, 255, 255])

            # HSV mask
            mask = cv2.inRange(hsv, lb, ub)

            # Removes the small patches in the image
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            # Dilation
            combined_dilated = cv2.dilate(opening, np.ones((31, 31), np.uint8), iterations=1)

            res = cv2.bitwise_and(image_ocv, image_ocv, mask=combined_dilated)

            # Find all the aruco markers in the image
            corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                         cameraMatrix=camera_matrix, distCoeff=camera_distortion)

            if np.all(ids is not None and ids != 0):  # ids[0] == id_to_find:
                axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1,
                                                                                                                    3)
                ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

                # Unpack the output, get only the first
                rvec, tvec = ret[0], ret[1]

                # Draw the detected marker and put a reference frame over it
                for i in range(0, ids.size):
                    aruco.drawAxis(image, camera_matrix, camera_distortion, rvec[i], tvec[i], 2.5)

                aruco.drawDetectedMarkers(image, corners)  # ids 3rd parameter

                # Print the ids of found markers
                str_id = ''
                for i in range(0, ids.size):
                    str_id += str(ids[i][0]) + ' '
                cv2.putText(image, "Id: " + str_id, (0, 25), font, 1, (0, 255, 255), 1, cv2.LINE_AA)

                for i in range(0, ids.size):
                    # Print the tag position in camera frame
                    str_position_first = "MARKER Position %d x=%.2f  y=%.2f  z=%.2f" % (
                        ids[0], tvec[0, 0, 0], tvec[0, 0, 1], tvec[0, 0, 2])
                    cv2.putText(image, str_position_first, (0, 80), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    if ids.size > 1:
                        str_position_second = "MARKER Position %d x=%.2f  y=%.2f  z=%.2f" % (
                            ids[1], tvec[1, 0, 0], tvec[1, 0, 1], tvec[1, 0, 2])
                        cv2.putText(image, str_position_second, (0, 120), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    # Obtain the rotation matrix tag->camera
                    R_ct_first = np.mat(cv2.Rodrigues(rvec[0, 0, :])[0])
                    R_tc_first = R_ct_first.T

                    # Get the attitude in terms of euler 321 (Needs to be flipped first)
                    roll_marker_first, pitch_marker_first, yaw_marker_first = rotationMatrixToEulerAngles(
                        R_flip * R_tc_first)

                    # Print the marker's attitude respect to camera frame
                    str_attitude_first = "MARKER Attitude %d r=%4.2f  p=%4.2f  y=%4.2f" % (ids[0],
                                                                                           math.degrees(
                                                                                               roll_marker_first),
                                                                                           math.degrees(
                                                                                               pitch_marker_first),
                                                                                           math.degrees(
                                                                                               yaw_marker_first))
                    cv2.putText(image, str_attitude_first, (0, 160), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    if ids.size > 1:
                        R_ct_second = np.mat(cv2.Rodrigues(rvec[1, 0, :])[0])
                        R_tc_second = R_ct_second.T
                        roll_marker_second, pitch_marker_second, yaw_marker_second = rotationMatrixToEulerAngles(
                            R_flip * R_tc_second)
                        str_attitude_second = "MARKER Attitude %d r=%4.2f  p=%4.2f  y=%4.2f" % (ids[1],
                                                                                                math.degrees(
                                                                                                    roll_marker_second),
                                                                                                math.degrees(
                                                                                                    pitch_marker_second),
                                                                                                math.degrees(
                                                                                                    yaw_marker_second))
                        cv2.putText(image, str_attitude_second, (0, 200), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # else:
                # Print 'No Ids' when no markers are found
                # cv2.putText(image, "No Ids", (0, 25), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            # depth_image = cv2.add(depth_image_ocv, res)
            # cv2.reprojectImageTo3D(depth_image_ocv, _3dImage, )

            # Draws a circle in the center of the image
            # cv2.circle(image, (x, y), 3, (255, 0, 255), 2)
            # cv2.circle(image_ocv, (i, j), 2, (255, 0, 255), 2)

            current_fps = zed.get_current_fps()
            print("Current framerate: ", current_fps)

            if not np.isnan(distance) and not np.isinf(distance):
                # Print the center position in camera frame
                center_pos = "Position: x=%d y=%d z=%.2f distance=%.2f" % (x, y, point_cloud_value[2], distance)
                cv2.putText(image, center_pos, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # print("Distance to Camera at ({0}, {1}): {2} cm".format(x, y, distance), end="\n")

                # Print the object(i, j) position in camera frame
                # object_pos = "Object Position: x=%d y=%d z=%.2f distance=%.2f" % (i, j, z1, distance1)
                # cv2.putText(image_ocv, object_pos, (0, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # print("Distance to Camera at ({0}, {1}, {2}): {3} cm".format(x1, y1, z1, distance1), end="\n")
            else:
                print("Can't estimate distance at this position.")
                print("Your camera is probably too close to the scene, please move it backwards.\n")

            # Imshow windows
            cv2.imshow("Image", image)
            cv2.imshow('Masked Image', mask)
            cv2.imshow('Opening', opening)
            cv2.imshow("combined_dilated", combined_dilated)
            cv2.imshow('Resulting Image', res)
            # cv2.imshow("Depth", depth)
            # cv2.imshow("Gray", gray)

            key = cv2.waitKey(10)

            process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")


if __name__ == "__main__":
    main()
