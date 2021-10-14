#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
using namespace std;

using namespace cv;

int main() { 

  // Read image from file
  Mat image = imread("mandrill.jpg", 1);
  Mat conv_image = imread("mandrill.jpg", 1);

  Mat kernel = Mat::ones(3, 3, CV_32F);
  kernel /= 9;
 
  // Threshold by looping through all pixels
  for(int y=1; y<image.rows-1; y++) {
   for(int x=1; x<image.cols-1; x++) {
     float pixel = 0.0;
     for(int i = -1; i<=1; i++) {
       for(int j= -1; j<=1; j++) {
         pixel += (double) (image.at<uchar>(y-i,x-j) * kernel.at<float>(i+1,j+1));
       }
     }
     conv_image.at<uchar>(y-1,x-1) = (uchar)(pixel);
} }

  //Save thresholded image
  imwrite("mandrill_convolve.jpg", conv_image);

  return 0;
}