// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;


void median(cv::Mat &input, int size, cv::Mat &output);

int main( int argc, char** argv )
{

    // LOADING THE IMAGE
    char* imageName = argv[1];

    Mat image;
    image = imread( imageName, 1 );

    if( argc != 2 || !image.data )
    {
    printf( " No image data \n " );
    return -1;
    }

    // CONVERT COLOUR, BLUR AND SAVE
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );

    Mat medCar;
    median(gray_image,5,medCar);

    imwrite( "median.jpg", medCar );

	//in-built function for median
	Mat medianOut;
	medianBlur(image, medianOut, 5);
    imwrite( "medianFunction.jpg", medianOut);

    return 0;
}

void median(cv::Mat &input, int size, cv::Mat &output)
{
	// intialise the output using the input
	output.create(input.size(), input.type());

	int radius = (size - 1)/2;


	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		radius, radius, radius, radius,
		cv::BORDER_REPLICATE );

	for(int i = 0; i<input.rows; i++) {
		for(int j=0; j<input.cols; j++) {
			vector<double> values;
			int counter = 0;
			for(int x = -radius; x <= radius; x++) {
				for(int y = -radius; y<= radius; y++) {
					int imagex = i + x + radius;
					int imagey = j + y + radius;
					values.push_back( (double) paddedInput.at<uchar>(imagex, imagey) );
				}
			}
			std::sort(values.begin(), values.end());
			output.at<uchar>(i,j) = (uchar) values[(values.size())/2 ];
		}
	}
}