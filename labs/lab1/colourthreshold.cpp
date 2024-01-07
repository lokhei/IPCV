//Basic Thresholding
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() { 

  // Read in image; 0 loads image as grayscale
  Mat image = imread("mandrillRGB.jpg", 1);

  // Threshold by looping through all pixels
  for (int y = 0; y<image.rows; y++) {
    for (int x = 0; x<image.cols; x++) {
        uchar blue = image.at<Vec3b>(y, x)[0];
        uchar green = image.at<Vec3b>(y, x)[1];
        uchar red = image.at<Vec3b>(y, x)[2];
        if (blue < 120 && red > 200 && green < 100 ){ //highlight nose
            image.at<Vec3b>(y, x)[0]= 255;
            image.at<Vec3b>(y, x)[1]= 255;
            image.at<Vec3b>(y, x)[2]= 255;
        }
        else {
           image.at<Vec3b>(y, x)[0]= 0;
            image.at<Vec3b>(y, x)[1]= 0;
            image.at<Vec3b>(y, x)[2]= 0;
        }
       
  } }

  //Save thresholded image
  imwrite("ColourThreshold.jpg", image);

  // using in-built function inRange
  Mat image2 = imread("mandrillRGB.jpg", 1);

  Mat mask;
  inRange(image2,  cv::Scalar(0, 0, 200),  cv::Scalar(120, 100, 255), mask);
  imwrite("ColourThresholdFunction.jpg", mask);

    
  return 0;
}