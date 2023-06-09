#pragma once
#include <tuple>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <ctime>
#include <time.h>   // this is needed for high resolution clock
#include <chrono>   // for high resolution clock
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING  // uncomment for Windows
#include <experimental/filesystem>    // uncomment for Windows

using namespace std;
using namespace cv;
//mespace fs = __fs::filesystem; // for MacOS
namespace fs = std::experimental::filesystem; // for Windows
using namespace fs;

//--------------------------------------------------------Feature Detector Class-----------------------------------------------------------------------------
 // Class to hold all the attribiutes realted to feature detector 
class FeatureDetector1 {
private:
    Ptr<FeatureDetector> sift = SIFT::create(); //SIFT
    Ptr<FeatureDetector> orb = ORB::create();   //ORB
    Ptr<FeatureDetector> brisk = BRISK::create();   //BRISK
    Ptr<FeatureDetector> akaze = AKAZE::create();   //AKAZE

public:
    //--------------------------------------------------------Keypoint Detection-----------------------------------------------------------------------------
    /** Method to detect keypoints in a given image
    * Preconditions: i) input Mat is a non-empty image
    *                ii) the method name is a valid feature detector technique
    * Postconditions: None
    * @return: A tuple of keypoints and descriptors in the input image
    */
    tuple<vector<KeyPoint>, Mat> detect1(Mat& img1, string method) {
        vector<KeyPoint> kp1;
        Mat des1;
        if (method == "SIFT") {
            sift->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = SIFT::create();
            extractor->compute(img1, kp1, des1);
        }
        else if (method == "ORB") {
            orb->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = ORB::create();
            extractor->compute(img1, kp1, des1);
        }
        else if (method == "BRISK") {
            brisk->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = BRISK::create();
            extractor->compute(img1, kp1, des1);
        }
        else if (method == "AKAZE") {
            akaze->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = AKAZE::create();
            extractor->compute(img1, kp1, des1);
        }
        return make_tuple(kp1, des1);
    }

    //--------------------------------------------------Matching Function for Keypoints----------------------------------------------------------------------
    /** Method to find the accurate matched between two sets of keypoints and descripotrs
    * Preconditions: i) the inputs keypoints and decriptors are corresponding to each other
    * Postconditions: None
    * @return: a vector containing tuples of matching points
    */
    vector<tuple<Point2f, Point2f, float>> match(vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, Mat& des1, Mat& des2) {
        vector<tuple<Point2f, Point2f, float>> result;

        for (int i = 0; i < des1.rows; i++) {
            Mat distance;
            reduce(abs(des2 - des1.row(i)), distance, 1, REDUCE_SUM);
            Mat sortedDist;
            sortIdx(distance, sortedDist, SORT_ASCENDING);

            int smallestIdx = sortedDist.at<int>(0);
            int secondSmallestIdx = sortedDist.at<int>(1);
            float smallestDistance = distance.at<float>(smallestIdx);
            float secondSmallestDistance = distance.at<float>(secondSmallestIdx);
            float ratio = smallestDistance / secondSmallestDistance;

            if (ratio < 0.8) {
                Point2f pt1 = kp1[i].pt;
                Point2f pt2 = kp2[smallestIdx].pt;
                result.push_back(make_tuple(pt2, pt1, ratio));
            }
        }

        return result;
    }

    //------------------------------------------------Detecting and Matching the Keypoints Detection---------------------------------------------------------
    /** Method to match the descriptors between two input images
    * Preconditions: method name is a valid feature detector
    * Postconditions: None
    * @return: a vector of matching correspondences
    */
    vector<pair<Mat, Mat>> detectAndMatch(Mat& img1, Mat& img2, string method) {

        vector<KeyPoint> kp1, kp2;
        Mat des1, des2;

        if (method == "SIFT") {
            sift->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = SIFT::create();
            extractor->compute(img1, kp1, des1);
            sift->detect(img2, kp2);
            Ptr<DescriptorExtractor> extractor1 = SIFT::create();
            extractor1->compute(img2, kp2, des2);
        }
        else if (method == "ORB") {
            orb->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = ORB::create();
            extractor->compute(img1, kp1, des1);
            orb->detect(img2, kp2);
            Ptr<DescriptorExtractor> extractor1 = ORB::create();
            extractor1->compute(img2, kp2, des2);
        }
        else if (method == "BRISK") {
            brisk->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = BRISK::create();
            extractor->compute(img1, kp1, des1);
            brisk->detect(img2, kp2);
            Ptr<DescriptorExtractor> extractor1 = BRISK::create();
            extractor1->compute(img2, kp2, des2);
        }
        else if (method == "AKAZE") {
            akaze->detect(img1, kp1);
            Ptr<DescriptorExtractor> extractor = AKAZE::create();
            extractor->compute(img1, kp1, des1);
            akaze->detect(img2, kp2);
            Ptr<DescriptorExtractor> extractor1 = AKAZE::create();
            extractor1->compute(img2, kp2, des2);
        }

        // Create a matcher
        vector<DMatch> matches;

        /*
        // For MacOS
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(des1, des2, matches);
        */

        // For Windows
        BFMatcher matcher(NORM_L2, true);
        matcher.match(des1, des2, matches);

        vector<pair<Mat, Mat>> correspondences;
        for (int i = 0; i < matches.size(); i++) {
            Point2f pt1 = kp1[matches[i].queryIdx].pt;
            Point2f pt2 = kp2[matches[i].trainIdx].pt;

            Mat pt1_homogeneous = (Mat_<double>(1, 3) << pt1.x, pt1.y, 1);
            Mat pt2_homogeneous = (Mat_<double>(1, 3) << pt2.x, pt2.y, 1);

            correspondences.push_back(make_pair(pt1_homogeneous, pt2_homogeneous));
        }

        return correspondences;
    }

    //------------------------------------------------Drawing the Circles for the keypoints------------------------------------------------------------------
    /** Method to encircle the keypoins in an input image
    * Preconditions: The input keypoints are valid for the input image
    * Postcondtions: Different circles get drawn on the input image
    * Assumptions: None
    */
    void drawCircle(Mat& image, vector<KeyPoint>& kp) {
        int H = image.rows;
        int W = image.cols;

        cout << "H=" << H << " " << "W=" << W << endl;

        for (int i = 0; i < kp.size(); i++) {
            Point2f pt = kp[i].pt;
            circle(image, Point(pt.x, pt.y), static_cast<int>(kp[i].size), Scalar(0, 255, 0), 1);
        }
    }
};