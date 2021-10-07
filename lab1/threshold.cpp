//Basic Thresholding
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() { 

  // Read in image; 0 loads image as grayscale
  Mat image = imread("mandrill.jpg", 0);

  // Threshold by looping through all pixels
  for (int y = 0; y<image.rows; y++) {
    for (int x = 0; x<image.cols; x++) {
      uchar pixel = image.at<uchar>(y, x);
      if (pixel>128) image.at<uchar>(y, x) = 255;
      else image.at<uchar>(y, x) = 0;
  } }


  //Save thresholded image
  imwrite("threshold.jpg", image);

  //Using OpenCv threshold function
  Mat thr;
  threshold( image, thr, 0, 255, THRESH_BINARY );
  imwrite("thrFunction.jpg", image);


  return 0;
}