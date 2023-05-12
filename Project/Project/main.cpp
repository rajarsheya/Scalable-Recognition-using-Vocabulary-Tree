//-----------------------------------------------------------------------------------------------------------------------------------------------------------
// 
// Enhanced Vocabulary Trees for Real-Time Object Recognition in Image and Video Streams
// Team members:  Arsheya Raj, Sugam Jaiswal, Josiah Zacharias
//
//-----------------------------------------------------------------------------------------------------------------------------------------------------------

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>   //added the OpenCV features2d module

using namespace std;
using namespace cv;

//--------------------------------------------------------------------ORB------------------------------------------------------------------------------------
/*
// Purpose: Create a ORB (Oriented FAST and Rotated BRIEF) Feature detector/descriptor and matcher.
// Preconditions: Both object.jpg and query.jpg exists in the code directory and is a valid JPG.
// Postconditions: The output for ORB is generated and displayed on the screen.
*/
void ORBfunction(Mat object, Mat query){
    // Initialize ORB detector
    Ptr<FeatureDetector> orb = ORB::create();

    // Detect keypoints
    std::vector<KeyPoint> ORBkeypoints1, ORBkeypoints2;
    orb->detect(object, ORBkeypoints1);
    orb->detect(query, ORBkeypoints2);

    // Compute descriptors
    Mat ORBdescriptors1, ORBdescriptors2;
    Ptr<DescriptorExtractor> extractor = ORB::create();
    extractor->compute(object, ORBkeypoints1, ORBdescriptors1);
    extractor->compute(query, ORBkeypoints2, ORBdescriptors2);

    // Match descriptors
    std::vector<DMatch> ORBmatches;
    Ptr<DescriptorMatcher> ORBmatcher = DescriptorMatcher::create("BruteForce-Hamming");
    ORBmatcher->match(ORBdescriptors1, ORBdescriptors2, ORBmatches);

    // Draw matches
    Mat ORBoutput;
    drawMatches(object, ORBkeypoints1, query, ORBkeypoints2, ORBmatches, ORBoutput);

    // Display the matches
    namedWindow("ORB Matches", WINDOW_NORMAL);
    resizeWindow("ORB Matches", ORBoutput.cols / 8, ORBoutput.rows / 8);
    imshow("ORB Matches", ORBoutput);
    waitKey(0);
}

//---------------------------------------------------------Driver Code - main--------------------------------------------------------------------------------
/*
// Purpose: Enhanced Vocabulary Trees for Real-Time Object Recognition in Image and Video Streams
// Preconditions: Both object.jpg and query.jpg exists in the code directory and is a valid JPG
// Postconditions: All the outputs are generated and displayed on the screen.
*/
 int main(int argc, char* argv[])
{
     // reading kittens1.jpg
     Mat object = imread("object.jpg");
     if (object.empty()){
         cout << "Error: Could not read object image file." << endl;
     }
     namedWindow("Object", WINDOW_NORMAL);
     imshow("Object", object);
     waitKey(0);
     
     // reading kittens2.jpg
     Mat query = imread("query.jpg");
     if (query.empty()){
         cout << "Error: Could not read query image file." << endl;
     }
     namedWindow("Query", WINDOW_NORMAL);
     imshow("Query", query);
     waitKey(0);
     
     ORBfunction(object, query);   // ORB (Oriented FAST and Rotated BRIEF)
     
     destroyAllWindows();
     return 0;
}