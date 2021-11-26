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
void detectAndDisplay( Mat frame );
void read_csv(string name, vector<Rect> &groundTruths, Mat frame );


/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	
	
	vector<Rect> groundTruths;
	read_csv(argv[1], groundTruths, frame);

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
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
/*
	int position; 


	while((position = line.find(delimiter)) != string::npos){
		tokens.push_back(line.substr(0, position));
		line.erase(0, position+1);
	}
	tokens.push_back(line);
*/
	return tokens;
	


}
void read_csv(string filePath, vector<Rect> &groundTruths, Mat frame ){

	ifstream file("labels.csv");
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



/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
