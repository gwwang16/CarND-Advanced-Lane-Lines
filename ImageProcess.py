import numpy as np
import cv2


class ImageProcess:
    """Image undistort, threshold and perspective"""

    def __init__(self, cali_pickle):
        # Load calibration data
        self.mtx = cali_pickle['mtx']
        self.dist = cali_pickle['dist']
        self.M = cali_pickle['M']
        self.src = cali_pickle['src']
        self.dst = cali_pickle['dst']
        # Initialization
        self.mag_threshold = (100, 250)
        self.mag_kernel = 3
        self.grad_kernel = 9
        self.dir_threshold = (0.7, np.pi / 2)
        self.s_threshold = (150, 250)

    def undistort(self, img):
        """Use the OpenCV undistort() function to remove distortion"""
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undistorted.astype(np.uint8)

    def mag_thresh(self, sobelx, sobely):
        """sobel magnitude threshold"""
        mag_sobel = np.sqrt(sobelx**2 + sobely**2)
        scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel > self.mag_threshold[0]) &
                   (scaled_sobel < self.mag_threshold[1])] = 1
        return mag_binary

    def dir_thresh(self, sobelx, sobely):
        """grad threshold"""
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
        dir_binary = np.zeros_like(dir_sobel)
        dir_binary[(dir_sobel > self.dir_threshold[0]) &
                   (dir_sobel < self.dir_threshold[1])] = 1

    def color_thresh(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # For yellow lane line
        yellow = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))
        # For white lane line
        sensitivity_1 = 68
        white = cv2.inRange(hsv, (0,0,255-sensitivity_1), (255,20,255))
        sensitivity_2 = 60
        hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        white_2 = cv2.inRange(hsl, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
        white_3 = cv2.inRange(img, (200,200,200), (255,255,255))
        # color_binary = np.zeros_like(hsv[:,:,0])
        # color_binary[(yellow==1) | (white==1) | (white_2==1) | (white_3==1)]=1
        color_binary = (yellow) | (white) | (white_2) | (white_3)
        color_binary[color_binary >= 1] = 1
        return color_binary

    def combine_thresh(self, undistort):
        """ Combine threshold binary images"""
        gray = cv2.cvtColor(undistort, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag_binary = self.mag_thresh(sobelx, sobely)
        # dir_binary = self.dir_thresh(sobelx, sobely)
        color_binary = self.color_thresh(undistort)
        # Stack channel to view their contributions
        combined_binary_color = np.dstack(
            (np.zeros_like(mag_binary), mag_binary, color_binary)) * 255
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(mag_binary)
        combined_binary[(color_binary == 1) | (mag_binary == 1)] = 1
        return combined_binary_color, combined_binary

    def perspective(self, combined_binary):
        """ Image perspective"""
        img_size = (combined_binary.shape[1], combined_binary.shape[0])
        warped = cv2.warpPerspective(
            combined_binary, self.M, img_size, flags=cv2.INTER_LINEAR)
        return warped
