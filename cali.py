import cv2
import numpy as np

def undistort(image, balance=0.0, dim2=None, dim3=None):
    dim1 = image.shape[:2][::-1]
    assert dim1[0] / dim1[1] == DIM[0] / DIM[1]
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]
    scaled_K[2][2] = 1.0
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 이미지를 저장합니다.
    output_path = "./cali/test/undistorted_image.jpg"
    cv2.imwrite(output_path, undistorted_img)

image = cv2.imread('./cali/test/1.jpg')

DIM=(2592, 1944)
K=np.array([[934.2740112139726, 0.0, 1300.0298126006094], [0.0, 933.9272687085421, 986.9224973823802], [0.0, 0.0, 1.0]])
D=np.array([[-0.04645933224643261], [0.0006314425785266855], [-0.006080866956130831], [0.0028393909411350607]])

undistort(image, balance=0.0, dim2=None, dim3=None)
