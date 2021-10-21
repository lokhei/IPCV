#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
using namespace std;

using namespace cv;

int main() { 

  // Read image from file
  Mat image = imread("mandrill.jpg", CV_LOAD_IMAGE_UNCHANGED);
  Mat conv_image = Mat::zeros(image.size(), image.type());


  Mat kernel = Mat::ones(3, 3, CV_32F);

  kernel /= 9; //normalise
  // float mykernel[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1}; //{0, -1, 0, -1, 5, -1, 0, -1, 0}
  // Mat kernel(3, 3, CV_32F, mykernel);

  
  // filter2D(image, conv_image, -1 , kernel); //inbuilt function

  // Threshold by looping through all pixels
  for(int y=1; y<image.rows; y++) {
    for(int x=1; x<image.cols; x++) {
      double pixel = 0.0;
      for(int i = -1; i<=1; i++) {
        for(int j= -1; j<=1; j++) {
          pixel += (double) (image.at<uchar>(y-i,x-j) * kernel.at<float>(i+1,j+1));
        }
      }
      conv_image.at<uchar>(y-1,x-1) = (uchar)(pixel);

  } }

  Mat filter;
  filter2D(image, filter, -1 , kernel);
  imwrite("filter2DMandrill.jpg", filter);




  //Save thresholded image
  imwrite("mandrill_convolve.jpg", conv_image);

  return 0;
}