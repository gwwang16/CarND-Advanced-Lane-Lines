import numpy as np
import cv2


class Lanes:
    """Detect and draw lane on image"""

    def __init__(self, cali_pickle):
        # Load calibration data
        self.mtx = cali_pickle['mtx']
        self.dist = cali_pickle['dist']
        self.M = cali_pickle['M']
        self.Minv = cali_pickle['Minv']
        self.src = cali_pickle['src']
        self.dst = cali_pickle['dst']
        # Initialization
        self.detection_state = False
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.y_max = 720
        self.nwindows = 12
        self.margin = 80
        self.minpix = 40  # minimum pixels to recenter window
        self.lane_width = 0
        self.left_fit_pre = [0,0,0]
        self.right_fit_pre = [0,0,0]
        # self.left_lane_idx = []  # left lane pixel indices list
        # self.right_lane_idx = []  # right lane pixel indices list

    def curvature(self):
        """Calculate lane curvatures"""
        ploty = self.lines['ploty']
        left_fitx = self.lines['left_fitx']
        right_fitx = self.lines['right_fitx']
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix,
                                 left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix,
                                  right_fitx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature (radius of curvature is in meters)
        left_num = (1 +(2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1])**2)**1.5
        left_den = np.absolute(2 * left_fit_cr[0])
        left_curverad = left_num / left_den
        right_num = (1 +(2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1])**2)**1.5
        right_den = np.absolute(2 * right_fit_cr[0])
        right_curverad = right_num / right_den
        # Now our radius of curvature is in meters
        return left_curverad, right_curverad

    def lane_offset(self):
        """Calculate current vechile position from center of the lane"""
        left_fit = self.lines['left_fit']
        right_fit = self.lines['right_fit']
        x_left = left_fit[0] * self.y_max**2 + \
            left_fit[1] * self.y_max + left_fit[2]
        x_right = right_fit[0] * self.y_max**2 + \
            right_fit[1] * self.y_max + right_fit[2]
        position = (x_right - x_left) / 2. + x_left - self.midpoint
        lane_width = x_right - x_left
        # Transfer pixel into meter
        position = position * self.xm_per_pix
        lane_width = lane_width * self.xm_per_pix
        self.lane_width = lane_width
        return position, lane_width

    def draw_lane(self, img, warped):
        """Draw lane onto the image"""
        ploty = self.lines['ploty']
        left_fitx = self.lines['left_fitx']
        right_fitx = self.lines['right_fitx']
        # Create image to draw lines
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast x and y points into usable format for cv2.fillpoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image
        warp_inv = cv2.warpPerspective(color_warp, self.Minv, self.img_size)
        # Combine the result with the original image
        out_img = cv2.addWeighted(img, 1, warp_inv, 0.3, 0)
        return out_img

    def find_lines_initial(self, warped):
        # Create an image to draw on and an image to show the selection window
        self.img_size = (warped.shape[1], warped.shape[0])
        out_img = np.dstack((warped, warped, warped)) * 255
        window_img = np.zeros_like(out_img)
        # Calculate histogram of the bottom half of warped image
        histogram = np.sum(warped[np.int(warped.shape[0] / 2):, :], axis=0)
        self.midpoint = np.int(histogram.shape[0] / 2)
        # Choose the left and right starting points
        leftx_base = np.argmax(histogram[:self.midpoint])
        rightx_base = np.argmax(histogram[self.midpoint:]) + self.midpoint
        # sliding windows height
        window_height = np.int(warped.shape[0] / self.nwindows)
        # Nonzero pixels index in the warped img
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current window position
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Left and right lane pixel indices lists
        left_lane_idx = []
        right_lane_idx = []

        for window in range(self.nwindows):
            # Left and right window boundaries
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = win_y_low + window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw left window
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 10)
            # Draw right window
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 10)
            # Identify nonzero pixels in window
            good_left_idx = ((nonzeroy >= win_y_low) &
                             (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) &
                             (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_idx = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) &
                              (nonzerox < win_xright_high)).nonzero()[0]
            # Append indices into lists
            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)
            if len(good_left_idx) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_idx]))
            if len(good_right_idx) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_idx]))

        # Concatenate indices arrays
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_idx]
        lefty = nonzeroy[left_lane_idx]
        rightx = nonzerox[right_lane_idx]
        righty = nonzeroy[right_lane_idx]
        # Fit a second order polynomial to each
        if (lefty.size==0) or (righty.size==0):
            left_fit = self.left_fit_pre
            right_fit = self.right_fit_pre
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)


        self.left_fit_pre = left_fit
        self.right_fit_pre = right_fit           

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        self.lines = {}
        self.lines['ploty'] = ploty
        self.lines['left_fitx'] = left_fitx
        self.lines['right_fitx'] = right_fitx
        self.lines['left_fit'] = left_fit
        self.lines['right_fit'] = right_fit
        out_img[nonzeroy[left_lane_idx], nonzerox[left_lane_idx]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_idx], nonzerox[right_lane_idx]]=[0, 0, 255]

        # Generate a polygon to illustrate the search window area
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        return result

    def find_lines(self, warped):
        """ Find lines on the warped image"""
        ploty = self.lines['ploty']
        left_fitx = self.lines['left_fitx']
        right_fitx = self.lines['right_fitx']
        left_fit = self.lines['left_fit']
        right_fit = self.lines['right_fit']
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_idx = ((nonzerox > (left_fit[0] *
                                      (nonzeroy**2) + left_fit[1] * nonzeroy +
                                      left_fit[2] - self.margin)) &
                         (nonzerox < (left_fit[0] *
                                      (nonzeroy**2) + left_fit[1] * nonzeroy +
                                      left_fit[2] + self.margin)))
        right_lane_idx = ((nonzerox >
                           (right_fit[0] *
                            (nonzeroy**2) + right_fit[1] * nonzeroy +
                            right_fit[2] - self.margin))
                          & (nonzerox <
                             (right_fit[0] *
                              (nonzeroy**2) + right_fit[1] * nonzeroy +
                              right_fit[2] + self.margin)))
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_idx]
        lefty = nonzeroy[left_lane_idx]
        rightx = nonzerox[right_lane_idx]
        righty = nonzeroy[right_lane_idx]
        # Fit a second order polynomial to each

        if (lefty.size==0) or (righty.size==0):
            left_fit = self.left_fit_pre
            right_fit = self.right_fit_pre
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit_pre = left_fit
        self.right_fit_pre = right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + \
            right_fit[1] * ploty + right_fit[2]
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped, warped, warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_idx], nonzerox[left_lane_idx]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_idx], nonzerox[right_lane_idx]] = [
            0, 0, 255
        ]

        self.lines = {}
        self.lines['ploty'] = ploty
        self.lines['left_fitx'] = left_fitx
        self.lines['right_fitx'] = right_fitx
        self.lines['left_fit'] = left_fit
        self.lines['right_fit'] = right_fit
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
        left_line_window2 = np.array([
            np.flipud(
                np.transpose(np.vstack([left_fitx + self.margin, ploty])))
        ])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
        right_line_window2 = np.array([
            np.flipud(
                np.transpose(np.vstack([right_fitx + self.margin, ploty])))
        ])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        return result
