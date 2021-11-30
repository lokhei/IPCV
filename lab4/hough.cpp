// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup


using namespace cv;

void sobel(cv::Mat &input, cv::Mat &deriv_x, cv::Mat &deriv_y,  cv::Mat &direction,  cv::Mat &magnitude);
void GaussianBlur(cv::Mat &input, int size,cv::Mat &blurredOutput);
void threshold(Mat &input, int threshold, Mat &output);
vector<vector<int> > hough_transform(Mat &input, Mat &mag, Mat &direction, int r_min, int r_max, double threshold);
void drawCircles(Mat &input, vector<vector<int> > circles);

int ***malloc3dArray(int dim1, int dim2, int dim3) {
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
        for (j = 0; j < dim2; j++) {
            array[i][j] = (int *) malloc(dim3 * sizeof(int));
        }

    }
    return array;
}




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

    Mat blurred;
    GaussianBlur(gray_image,7,blurred);

    //sobel
    Mat deriv_x(image.size(),  CV_32FC1);
    Mat deriv_y(image.size(),  CV_32FC1);
    Mat magnitude(image.size(),  CV_32FC1);
    Mat direction(image.size(), CV_32FC1);   
    sobel(blurred, deriv_x, deriv_y, direction, magnitude);


    //normalise result
    Mat r_deriv_x(image.size(), CV_8UC1);
    Mat r_deriv_y(image.size(), CV_8UC1);
    Mat r_magnitude(image.size(), CV_8UC1, Scalar(0));
    Mat r_direction(image.size(), CV_8UC1, Scalar(0));
    normalize(deriv_x,r_deriv_x,0,255,NORM_MINMAX, CV_8UC1);
    normalize(deriv_y, r_deriv_x, 0,255,NORM_MINMAX, CV_8UC1);
    normalize(magnitude,r_magnitude,0,255,NORM_MINMAX);
    normalize(direction,r_direction,0,255,NORM_MINMAX);

    imwrite( "SobelXDirection.jpg", r_deriv_x );
    imwrite( "SobelYDirection.jpg", r_deriv_x );
    imwrite( "SobelMag.jpg", r_magnitude );
    imwrite( "SobelDir.jpg", r_direction );


    //threshold
    Mat thres_mag = imread("SobelMag.jpg", 1);
    Mat gray_thres;
    cvtColor( thres_mag, gray_thres, CV_BGR2GRAY );
    threshold(gray_thres, 50, thres_mag);
    // vector<vector<int> > circles = hough_transform(image, thres_mag, direction, 10, 100, 20);
    vector<vector<int> > circles = hough_transform(image, thres_mag, direction, 5, 70, 14);

    drawCircles(image, circles);


    return 0;
}


void sobel(cv::Mat &input, cv::Mat &deriv_x, cv::Mat &deriv_y, cv::Mat &direction, cv::Mat &magnitude) {

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
    
    for ( int i = 0; i < input.rows; i++ ) {	
        for( int j = 0; j < input.cols; j++ ) {
            float sum_x = 0.0;
            float sum_y = 0.0;
            for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
                for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    float imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
                    float kernel_x = kernelX.at<float>( kernelx, kernely );
                    float kernel_y = kernelY.at<float>( kernelx, kernely );

                    sum_x += imageval * kernel_x;
                    sum_y += imageval * kernel_y;
                }
            }
            deriv_x.at<float>(i, j) = (float) sum_x;
            deriv_y.at<float>(i, j) = (float) sum_y;
            magnitude.at<float>(i, j) = (float) sqrt((sum_y*sum_y) + (sum_x*sum_x));
            direction.at<float>(i, j) = (float) atan2(sum_y, sum_x);
        }
    }
}

vector<vector<int> > hough_transform(Mat &image, Mat &mag,  Mat &direction, int r_min, int r_max, double threshold) {

    int ***hough_space = malloc3dArray(mag.rows, mag.cols, r_max);
    for (int i = 0; i < mag.rows; i++) {
        for (int j = 0; j < mag.cols; j++) {
            for (int r = 0; r < r_max; r++) {
                hough_space[i][j][r] = 0;
            }
        }
    }


    for (int x = 0; x < mag.rows; x++) {
        for (int y = 0; y < mag.cols; y++) {
            if(mag.at<uchar>(x,y) == 255) { //determine pixels with strongest gradient magnitude
                for (int r = 0; r < r_max; r++) {
                    int xc = int(r * sin(direction.at<float>(x,y)));
                    int yc = int(r * cos(direction.at<float>(x,y)));

                    int a = x - xc;
                    int b = y - yc;
                    int c = x + xc;
                    int d = y + yc;
                    //increment if satisfy equations
                    if(a >= 0 && a < mag.rows && b >= 0 && b < mag.cols) {
                        hough_space[a][b][r] += 1;
                    }
                    if(c >= 0 && c < mag.rows && d >= 0 && d < mag.cols) {
                        hough_space[c][d][r] += 1; 
                    }
                }
            }
        }
    }

    Mat hough_output(mag.rows, mag.cols, CV_32FC1);

    //sum  values of the radius dimension
    for (int x = 0; x < mag.rows; x++) {
        for (int y = 0; y < mag.cols; y++) {
            for (int r = r_min; r < r_max; r++) {
                hough_output.at<float>(x,y) += hough_space[x][y][r];
            }
        }
    }

    //normalize
    Mat hough_norm(mag.rows, mag.cols, CV_8UC1);
    normalize(hough_output, hough_norm, 0, 255, NORM_MINMAX);
    imwrite( "hough.jpg", hough_norm );



    vector<vector<int> > circles;
	for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
			bool test_pass = true;

            int currentMax = 0;
            int maxRadius = 0;

            // find max radius
            for (int r = r_min; r < r_max; r++) {
				if(hough_space[x][y][r] > currentMax) {
					currentMax = hough_space[x][y][r];
                    maxRadius = r;
				}
            }
			
			for(int i = 0; i < circles.size(); i++) {
				vector<int> circle = circles[i];
				int xc = circle[0];
				int yc = circle[1];
				int rc = circle[2];

				if(!(pow((x-xc),2) + pow((y-yc),2) > pow(rc,2))) {
					test_pass = false;
				}
			}
			if(hough_space[x][y][maxRadius] > threshold && test_pass) {
				vector<int> circle;
				circle.push_back(x);
				circle.push_back(y);
				circle.push_back(maxRadius);
				circles.push_back(circle);
			}
        }
    }

	std::cout << "circles: " << circles.size() << std::endl;


    imwrite( "detectedCircles.jpg", image );

    return circles;
}

void drawCircles(Mat &input, vector<vector<int> > circles) {

	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[1], c[0]);
		circle(input, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		int radius = c[2];
		circle(input, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}

	imwrite("detectedCircles.jpg", input);

}



void threshold(Mat &input, int threshold, Mat &output) {
    assert(threshold >= 0 && threshold <= 255);
    output.create(input.size(), input.type());
    for(int i = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++) {
            int val = (int) input.at<uchar>(i, j);
            if(val > threshold) {
                output.at<uchar>(i,j) = (uchar) 255;
            } else {
                output.at<uchar>(i,j) = (uchar) 0;
            }
        }
    }
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput){
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

    // now we can do the convolution
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
