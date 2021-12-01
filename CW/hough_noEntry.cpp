/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
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
void detectAndDisplay( Mat frame, vector<Rect> &faces );
void read_csv(string name, vector<Rect> &groundTruths, Mat frame );
float iou(Rect truth, Rect detected);
float calcFaceCount(vector<Rect> &faces, vector<Rect> &groundTruths, float iou_thres);
float calcTPR(int faceCount, int truthsSize);
float f1Score(int faceSize, int truthSize, int faceCount);
vector<string> splitString(string s, string delimiter);

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ){
    const char* imageName = argv[1];

	// 1. Read Input Image
	Mat frame = imread(imageName, CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // read in and draw groundTruth
    vector<Rect> groundTruths;
	read_csv(imageName, groundTruths, frame);

	// 3. Detect Faces and Display Result
	vector<Rect> faces;
	detectAndDisplay(frame, faces);
	std::cout << "Number of Faces: " << faces.size() << std::endl;

    

	//faces successfully detected
	float faceCount = calcFaceCount(faces, groundTruths, 0.4);
	std::cout << "Number of successful faces : " << faceCount << endl;
    // TPR
	float TPR = calcTPR(faceCount, groundTruths.size());
	std::cout << "TPR: " << TPR << endl;
    // F1 score
	float f1 = f1Score(faces.size(), groundTruths.size(), faceCount);
	std::cout << "F1 Score " << f1 << endl;

	// imwrite( "detected.jpg", frame );




    Mat image = imread( imageName, 1 );

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
    Mat r_magnitude(image.size(), CV_8UC1, Scalar(0));
    normalize(magnitude,r_magnitude,0,255,NORM_MINMAX);

    //threshold magnitude
    Mat thres_mag = imread("SobelMag.jpg", 1);
    Mat gray_thres;
    cvtColor( thres_mag, gray_thres, CV_BGR2GRAY );
    threshold(gray_thres, 50, thres_mag);

    vector<string> fileName = splitString(imageName, "/");
	string houghOut = "hough/hough_" + splitString(fileName[fileName.size()-1], ".")[0] +".jpg";

    //perform hough transform
    vector<vector<int> > circles = hough_transform(frame, thres_mag, direction, 5, 100, 20, houghOut);
    // vector<vector<int> > circles = hough_transform(image, thres_mag, direction, 5, 70, 14);

    drawCircles(frame, circles);

	// Save Result Image
	string imageOut = "hough/detected_" + splitString(fileName[fileName.size()-1], ".")[0] +".jpg";
	imwrite(imageOut, frame);


	return 0;
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

	// imwrite("detected.jpg", input);

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
	float height = min(truth.y + truth.height, detected.y + detected.height) - max(truth.y, detected.y);
	float width = min(truth.x + truth.width, detected.x + detected.width) - max(truth.x, detected.x);
    if (height < 0 || height < 0) return 0;
	float intersect = height*width;

	float unionArea = truth.height*truth.width + detected.height*detected.width - intersect;

	return intersect/unionArea;
}

float calcFaceCount(vector<Rect> &faces, vector<Rect> &groundTruths, float iou_thres){
	int faceCount = 0;
	for (int j = 0; j < groundTruths.size(); j++){

		for( int i = 0; i < faces.size(); i++ ){
			if (iou(groundTruths[j], faces[i]) > iou_thres){
				faceCount ++;
				break;
			}
		}
	}
	return faceCount;
}



float calcTPR(int faceCount,  int truthsSize){
	if (truthsSize == 0) return 0;
	return faceCount/float(truthsSize);
}

float f1Score(int faceSize, int truthSize, int faceCount){
	float fp = faceSize - faceCount;
	float fn = truthSize - faceCount;
	float f1 = faceCount/(faceCount+0.5*(fp+fn));

	return f1;
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, vector<Rect> &faces){
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );


	// 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
