/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - hough_noEntry.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;

/** Function Headers */
void sobel(cv::Mat &input, cv::Mat &deriv_x, cv::Mat &deriv_y,  cv::Mat &direction,  cv::Mat &magnitude);
void GaussianBlur(cv::Mat &input, int size,cv::Mat &blurredOutput);
void threshold(Mat &input, int threshold, Mat &output);
vector<vector<int> > hough_transform(Mat &input, Mat &mag, Mat &direction, int r_min, int r_max, double threshold, string fileName);
void drawCircles(Mat &input, vector<vector<int> > circles);
void detectAndDisplay( Mat frame, Mat &drawImage, vector<Rect> &signs );
void read_csv(string name, vector<Rect> &groundTruths, Mat frame );
float iou(Rect truth, Rect detected);
float calcSignCount(vector<Rect> &signs, vector<Rect> &groundTruths, float iou_thres);
float calcTPR(int signCount, int truthsSize);
float f1Score(int signSize, int truthSize, int signCount);
vector<string> splitString(string s, string delimiter);
vector<Rect> drawDetected(Mat image, vector<Rect> &signs, vector<vector<int> > &houghCircle, float iou_thres);

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ){
    const char* imageName = argv[1];
    vector<string> fileName = splitString(imageName, "/");

    Mat image = imread( imageName, CV_LOAD_IMAGE_COLOR );

   
	// Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	


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
    Mat r_magnitude(image.size(), CV_8UC1);
    normalize(magnitude,r_magnitude,0,255,NORM_MINMAX);


    imwrite( "SobelMag.jpg", r_magnitude );
	
    //threshold
    Mat thres_mag = imread("SobelMag.jpg", 1);
    Mat gray_thres;
    cvtColor( thres_mag, gray_thres, CV_BGR2GRAY );
    threshold(gray_thres, 50, thres_mag);

	string thresName = "hough/thresMag_" + splitString(fileName[fileName.size()-1], ".")[0] +".jpg"; 
    imwrite( thresName, thres_mag );  //save thresholded mag image



    


    // Detect signs and Display Result
    Mat drawImage = imread( imageName, CV_LOAD_IMAGE_COLOR );
	vector<Rect> detected;
	detectAndDisplay(image, drawImage, detected);
    string detectedOut = "hough/detected_" + splitString(fileName[fileName.size()-1], ".")[0] +".jpg";
	imwrite(detectedOut, drawImage); //save viola-jones detected image
	std::cout << "Number of detected Signs: " << detected.size() << std::endl; // from viola Jones

    
    //find max size of bounding box
    float max_r = 0;
	for (int i = 0; i < detected.size(); i++){
        float sign_maxR =  sqrt(pow(detected[i].width,2)+ pow(detected[i].height,2));
        if (sign_maxR > max_r) max_r = sign_maxR;
    }


    
	string houghOut = "hough/hough_" + splitString(fileName[fileName.size()-1], ".")[0] +".jpg";

    vector<vector<int> > circles = hough_transform(image, thres_mag, direction, 16, max_r, 15, houghOut);
    
    Mat circleImage = imread( imageName, CV_LOAD_IMAGE_COLOR ); 

    drawCircles(circleImage, circles);
    string circleFileName = "hough/circles_" + splitString(fileName[fileName.size()-1], ".")[0] +".jpg";
	imwrite(circleFileName, circleImage); //save hough circles image

    // read in and draw groundTruth
    vector<Rect> groundTruths;
	read_csv(imageName, groundTruths, image);

    vector<Rect> filteredSignCount = drawDetected(image, detected, circles, 0.4); //filter viola with the circles
    std::cout << "Number of filtered detections : " << filteredSignCount.size() << endl; //circles with viola jones

	// Save Result Image
	string imageOut = "hough/filtered_" + splitString(fileName[fileName.size()-1], ".")[0] +".jpg";
	imwrite(imageOut, image); //save filtered image with ground truth

    //signs successfully detected
	float signCount = calcSignCount(filteredSignCount, groundTruths, 0.4); //compare filtered with ground truth
	std::cout << "Number of successful filtered detections : " << signCount << endl;
    // TPR
	float TPR = calcTPR(signCount, groundTruths.size());
	std::cout << "TPR: " << TPR << endl;
    // F1 score
	float f1 = f1Score(filteredSignCount.size(), groundTruths.size(), signCount);
	std::cout << "F1 Score " << f1 << endl;


	return 0;
}

Rect convCircle2Rect(vector<int> circle){
    return Rect(Point(circle[1]+circle[2], circle[0]+circle[2]),Point(circle[1]-circle[2], circle[0]-circle[2]));
}


vector<Rect> drawDetected(Mat image, vector<Rect> &signs,vector<vector<int> > &houghCircle, float iou_thres){
    vector<Rect> rects;
	for (int j = 0; j < houghCircle.size(); j++){
        float maxIOU = 0;
        Rect maxRect;
        Rect houghRect = convCircle2Rect(houghCircle[j]);
		for( int i = 0; i < signs.size(); i++ ){
            float iouVal = max(iou(houghRect, signs[i]), iou(signs[i], houghRect));
            if (maxIOU < iouVal){
                maxIOU = iouVal;
                maxRect = signs[i];
            }
		}

        if (maxIOU > iou_thres){
            rectangle(image, Point(maxRect.x, maxRect.y), Point(maxRect.x + maxRect.width, maxRect.y + maxRect.height), Scalar( 0, 255, 0 ), 2);
            rects.push_back(maxRect);
        }
	}

    return rects;

}





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

vector<vector<int> > hough_transform(Mat &image, Mat &mag,  Mat &direction, int r_min, int r_max, double threshold, string fileName) {

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
    imwrite( fileName, hough_norm );



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


    // imwrite( "detectedCircles.jpg", image );

    return circles;
}


void drawCircles(Mat &input, vector<vector<int> > circles) {

	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[1], c[0]);
		circle(input, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		int radius = c[2];
		circle(input, center, radius, Scalar(128, 0, 128), 2, 8, 0);
	}
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

    // now we can do the Colution
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


vector<string> splitString(string s, string delimiter){
	vector<string> tokens;

	int start = 0;
	int end = s.find(delimiter);
	while (end != -1){
		tokens.push_back(s.substr(start,end-start));
		start = end + delimiter.size();
		end = s.find(delimiter, start);
	}
	tokens.push_back(s.substr(start,end-start));
	return tokens;
}


vector<vector<int> > hough_circles(Mat &input, int r_min, int r_max, double threshold, Mat &direction, string imageNum) {

	int ***hough_space = malloc3dArray(input.rows, input.cols, r_max);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            for (int r = 0; r < r_max; r++) {
                hough_space[i][j][r] = 0;
            }
        }
    }

	//-- Make the hough space -- //
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
			if(input.at<uchar>(y,x) == 255) {
				for (int r = 0; r < r_max; r++) {
					int yc = int(r * sin(direction.at<float>(y,x)));
					int xc = int(r * cos(direction.at<float>(y,x)));

					int a = x - xc;
					int b = y - yc;
					if(a >= 0 && a < input.cols && b >= 0 && b < input.rows) {
						hough_space[b][a][r] ++;
					}

					int c = x + xc;
					int d = y + yc;
					if(c >= 0 && c < input.cols && d >= 0 && d < input.rows) {
						hough_space[d][c][r] ++;
					}
				}
			}
        }
    }

	Mat hough_output(input.rows, input.cols, CV_32FC1);
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            for (int r = r_min; r < r_max; r++) {
                hough_output.at<float>(y,x) += hough_space[y][x][r];
            }
 
        }
    }

    imwrite( "goundTruth_NoEntry_Hough/detected_NE_" + imageNum + "/hough_space.jpg", hough_output );

	//--- get the circles ---//
	vector<vector<int> > circles;
	for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
			bool test_pass = true;
		
			int max_r = 0;
			int currentMax = 0;
			
			for (int r = r_min; r < r_max; r++) {
				if ( hough_space[y][x][r] > currentMax) {
					currentMax = hough_space[y][x][r];
					max_r = r;
				}	
			}
				
			for(int i = 0; i < circles.size(); i++) {
				vector<int> circle = circles[i];
				int xs = circle[0];
				int ys = circle[1];
				int rs = circle[2];

				//equation of a circle (x'-x)^2+(y'-y)^2 = r^2 were x & y are center
				if(!(pow((xs-x),2) + pow((ys-y),2) > pow(rs,2))) {
					test_pass = false;
				}
			}
			if(hough_space[y][x][max_r] > threshold && test_pass) {
				vector<int> circle;
				circle.push_back(x);
				circle.push_back(y);
				circle.push_back(max_r);
				circles.push_back(circle);
			}
        }
    }
	return circles;
}
void read_csv(string filePath, vector<Rect> &groundTruths, Mat frame ){

	ifstream file("noEntry_labels.csv");
	string line;

	vector<string> fileName = splitString(filePath, "/");

	string imageName = splitString(fileName[fileName.size()-1], ".")[0];



	while(getline(file, line)){

		vector<string> col = splitString(line, ",");
		if (splitString(col[0], ".")[0] == imageName) {
			groundTruths.push_back(Rect(atoi(col[1].c_str()), atoi(col[2].c_str()), atoi(col[3].c_str()), atoi(col[4].c_str())));
		}
	}

	for (int i = 0; i < groundTruths.size(); i++){
		rectangle(frame, Point(groundTruths[i].x, groundTruths[i].y), Point(groundTruths[i].x + groundTruths[i].width, groundTruths[i].y + groundTruths[i].height), Scalar( 0, 0, 255 ), 2);
	}
	
	file.close();
}

float iou(Rect truth, Rect detected){
    return (truth & detected).area() / (float) (detected | truth).area();
}

float calcSignCount(vector<Rect> &signs, vector<Rect> &groundTruths, float iou_thres){
	int signCount = 0;
	for (int j = 0; j < groundTruths.size(); j++){

		for( int i = 0; i < signs.size(); i++ ){
			if (iou(groundTruths[j], signs[i]) > iou_thres){
				signCount ++;
				break;
			}
		}
	}
	return signCount;
}



float calcTPR(int signCount,  int truthsSize){
	if (truthsSize == 0) return 0;
	return signCount/float(truthsSize);
}

float f1Score(int signSize, int truthSize, int signCount){
	float fp = signSize - signCount;
	float fn = truthSize - signCount;
	float f1 = signCount/(signCount+0.5*(fp+fn));

	return f1;
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, Mat &drawImage,vector<Rect> &signs){
	Mat frame_gray;
	Mat blurred;


	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );


	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, signs, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

    for( int i = 0; i < signs.size(); i++ )
	{
		rectangle(drawImage, Point(signs[i].x, signs[i].y), Point(signs[i].x + signs[i].width, signs[i].y + signs[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
