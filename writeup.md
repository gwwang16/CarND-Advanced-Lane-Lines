## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/undistorted_test1.png "Road Undistorted"
[image7]: ./output_images/pespective.png "Pespective"
[image3]: ./output_images/binary_result.png "Binary "
[image4]: ./output_images/warped.png "Warp"
[image8]: ./output_images/histogram.png "Historgram"
[image5]: ./output_images/line_fitting.png "Fit Visual"
[image6]: ./output_images/result.png "Output"
[video1]: ./project_video_output.mp4 "Video"
[image9]: ./output_images/video_screenshot.png "Video output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is contained in `Camera_calibration.ipynb`

The steps contain:

- obtain `object points` and `corners` from chessboard pictures using `cv2.findChessboardCorners()`

- compute camera matrix and distortion coefficients using `cv2.calibrateCamera()` 

- apply distortion correction to image using `cv2.undistort()`

The following result is obtained

![alt text][image1]

### Pipeline (single images)

There are two main python files: `ImageProcess.py` and `LaneFinding.py`.

 `ImageProcess.py` is used for image processing contains distortion correction, extracting lane line pixels and image perspective.

`LaneFinding.py` is used for lane lines fitting, curvature calculating and draw lane, etc.

#### 1. Provide an example of a distortion-corrected image.

apply the distortion correction to one of the test images
![alt text][image2]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and sobel magnitude thresholds to generate a binary image.  I also tried gradient threshold and absolute sobel methods, but didn't find good result.  The color threshold method adopted the code was presented in slack channel by @kylesf. This code can be found in `combine_thresh()` in `ImageProcess.py`

Here's an example of my output for this step

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform is `perspective()` in the file `ImageProcess.py` , it uses transform matrix, which is obtained during camera calibration using `cv2.getPerspectiveTransform(src, dst)`

The following source and destination points are used

|  Source   | Destination |
| :-------: | :---------: |
| 200, 720  |  350, 720   |
| 563, 470  |   350, 0    |
| 723, 470  |   980, 0    |
| 1130, 720 |  980, 720   |

I verified this perspective transform on a straight line image, the lines is parallel on the warped image, it means the perspective transform is reasonable.

![alt text][image7]

Then, the perspective transform is used into previous test image, the following result is obtained
![alt text][image4]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

We need locate the lane lines at first, the code is `find_lines_initial()` in `LaneFinding.py`.

 I split the image into 12 slices,  then

- determine left and right lines starting point of the bottom slice based on histogram result.

![alt text][image8]


- set a sliding window with `2*margin` pixels width and  `np.int(warped.shape[0]/nwindows)`  height around left and right starting points.

- set left and right lines starting points of the next slice as mean value of nonzero pixel indexes in the previous window

- repeat 2-3 steps to the last slice

- fit lane lines with 2 order polynomial using the above points of left and right, respectively.

  The result is the following 

  ![alt text][image5]



Once we know where the lines are, the sliding windows step can be skipped, the function `find_lines()` is used here. The sliding window is replaced by the adjacent domains around the previous fitting lines.



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated by `curvature()` and the lane offset is calculated by `lane_offset()` in `LaneFinding.py`

##### 1) Curvature

We have obtained the fitting polynomial 

f(y) = Ay^2 + By + C

>
> fitting for f(y), rather than f(x), because the lane lines in the warped image are near vertical and may have the same x value for more than one y value.
>

The radius of curvature at any point x of the function x=f(y) is given as follows:

R_{curve} = (1+(2Ay+B)^2)^(3/2) / |2A|

The y values of image increase from top to bottom, so if we wanted to measure the radius of curvature closest to vehicle, we should evaluate the formula above at the y value corresponding to the bottom of  image.

Then, we should transfer the curvature from pixels into meters by multiplying coefficients, which can be calculated by measuring out the physical lane in the field of view of the camera. But rough estimated coefficients are used in this project

```
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

##### 2) Lane offset

The position of the vehicle with respect to center can be calculated with the relative position between image midpoint and lane midpoint. The code is the following


```
x_left = left_fit[0] * self.y_max**2 + left_fit[1] * self.y_max + left_fit[2]
x_right = right_fit[0] * self.y_max**2 + right_fit[1] * self.y_max + right_fit[2]
offset = (x_right - x_left) / 2. + x_left - self.midpoint
lane_width = x_right - x_left
# Transfer pixel into meter
offset = offset * self.xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I used `draw_lane()` in file `LaneFinding.py` to draw the identified lane on the original image. 

Here is an example on a test image:
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4) 
![alt text][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I implement a lane finding algorithm in this project, it performs well on the project video, but not well on the challenge video and fails on the hard challenge video. There are many points need to be improved further, but I have no time for this course, have to move into the next step. I hope I could improve it further in the future.

- The lane detecting algorithm is not robust enough for noise, such as shadow and blur lane lines, especially for the trees in the harder challenge video. The image filtering method need to be improved.
- The lane lines should be parallel for most of time, judging statement is preferred to select one better fitting line.