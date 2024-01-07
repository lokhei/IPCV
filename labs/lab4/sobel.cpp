// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup


using namespace cv;

void sobel(cv::Mat &input);
void normalise(Mat &input, string name);
void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);



int main( int argc, char** argv ) {

	// LOADING THE IMAGE
	char* imageName = argv[1];

	Mat image;
	image = imread( imageName, 1 );

	if( argc != 2 || !image.data ){
		printf( " No image data \n " );
		return -1;
	}

	// CONVERT COLOUR, BLUR AND SAVE
	Mat gray_image;
	cvtColor( image, gray_image, CV_BGR2GRAY );
	// Mat blurred;
    // GaussianBlur(gray_image,8,blurred);

	sobel(gray_image);


	return 0;
}

void sobel(cv::Mat &input) {

	Mat deriv_x;
	Mat deriv_y;
	Mat magnitude;
	Mat direction;
	deriv_x.create(input.size(), input.type());
	deriv_y.create(input.size(), input.type());
	magnitude.create(input.size(), input.type());
	direction.create(input.size(), input.type());
	// intialise the output using the input


	float kernel[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  	Mat kernelX(3, 3, CV_32F, kernel);

	Mat kernelY = kernelX.t();
	

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernelX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernelY.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convolution
	for ( int i = 0; i < input.rows; i++ ){	
		for( int j = 0; j < input.cols; j++ ){
			double sum_x = 0.0;
			double sum_y = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ){
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ){
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernel_x = kernelX.at<double>( kernelx, kernely );
					double kernel_y = kernelY.at<double>( kernelx, kernely );

					// do the multiplication
					sum_x += imageval * kernel_x;
					sum_y += imageval * kernel_y;
				}
			}
			// set the output value as the sum of the convolution
			deriv_x.at<uchar>(i, j) = (uchar) sum_x;
			deriv_y.at<uchar>(i, j) = (uchar) sum_y;
			magnitude.at<uchar>(i, j) = (uchar) sqrt((sum_y*sum_y) + (sum_x*sum_x));
			direction.at<uchar>(i, j) = (uchar) atan2(sum_y,sum_x);
		}
	}


	// imwrite( "coin_x1.jpg", deriv_x );
	// imwrite( "coin_y1.jpg", deriv_y );
	// imwrite( "coin_mag1.jpg", magnitude );
	// imwrite( "coin_dir1.jpg", direction );


	normalise(deriv_x, "x");
	normalise(deriv_y, "y");
	normalise(magnitude, "mag");
	normalise(direction, "dir");
}

void normalise(Mat &input, string name) {
	double min; 
	double max; 
	minMaxLoc( input, &min, &max );

	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			double val = (double) input.at<uchar>(i, j);
			input.at<uchar>(i,j) = (uchar) (val - min)*((255)/max-min);
		}
	}
	imwrite( "coin_" + name + ".jpg", input );
}



void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}

