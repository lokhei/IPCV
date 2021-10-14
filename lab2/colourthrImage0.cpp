/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: RGB explicit thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() { 

  // Read image from file
    Mat image = imread("mandrill0.jpg", 1);


    // Threshold by looping through all pixels
    for(int y=0; y<image.rows; y++) {
        for(int x=0; x<image.cols; x++) {
            uchar pixelBlue = image.at<Vec3b>(y,x)[0];
            uchar pixelGreen = image.at<Vec3b>(y,x)[1];
            uchar pixelRed = image.at<Vec3b>(y,x)[2];


        // swap colour values
            image.at<Vec3b>(y,x)[0]=pixelRed;
            image.at<Vec3b>(y,x)[1]=pixelBlue;
            image.at<Vec3b>(y,x)[2]=pixelGreen;

        } }

   
    imwrite("reconstructed.jpg", image);
  
    return 0;
}