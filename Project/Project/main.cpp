#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
				
int main(int argc, char* argv[])
{	
	Mat image = imread("pippy.jpg");
		
	namedWindow("Original Image");
	imshow("Original Image", image);
	waitKey(0);

	return 0;
}
