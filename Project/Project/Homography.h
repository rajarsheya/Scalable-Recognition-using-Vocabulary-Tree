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


//-----------------------------------------------------------Homography Function-----------------------------------------------------------------------------
/** Function to calculate homography matrix relating the transformation between two images
* Preconditions: The input vector correspondences is non-empty
* Postconditions: The matrix H is normalized
* @return: a 3*3 matrix H
*/
Mat homography(vector<pair<Mat, Mat>> correspondences) {
    int n = (int)correspondences.size(); // number of points
    Mat A = Mat::zeros(2 * n, 9, CV_64F);
    for (int i = 0; i < n; i++) {
        Mat pt1 = correspondences[i].first;
        Mat pt2 = correspondences[i].second;
        A.at<double>(2 * i, 0) = pt1.at<double>(0);
        A.at<double>(2 * i, 1) = pt1.at<double>(1);
        A.at<double>(2 * i, 2) = 1;
        A.at<double>(2 * i, 6) = -pt1.at<double>(0) * pt2.at<double>(0);
        A.at<double>(2 * i, 7) = -pt1.at<double>(0) * pt2.at<double>(1);
        A.at<double>(2 * i, 8) = -pt2.at<double>(0);
        A.at<double>(2 * i + 1, 3) = pt1.at<double>(0);
        A.at<double>(2 * i + 1, 4) = pt1.at<double>(1);
        A.at<double>(2 * i + 1, 5) = 1;
        A.at<double>(2 * i + 1, 6) = -pt2.at<double>(1) * pt1.at<double>(0);
        A.at<double>(2 * i + 1, 7) = -pt2.at<double>(1) * pt1.at<double>(1);
        A.at<double>(2 * i + 1, 8) = -pt2.at<double>(1);
    }
    Mat U, S, Vt;
    SVD::compute(A, S, U, Vt);
    Mat H = Vt.row(Vt.rows - 1).reshape(0, 3); // last row of Vt
    return H;
}

//------------------------------------------Number of rounds needed for RANSAC to have P chance to success---------------------------------------------------
/** Method to calculate the number of rounds needed for succes
* Preconditions: p is the probability that the sample match is inliner
*                k is the number of matches needed for one round
*                P is the probaility of sucess after 5 rounds
* Postconditions: None
* @return: the number of rounds S
*/
int rounds(double p, int k, double P) {
    double S = log(1 - P) / log(1 - pow(p, k));
    cout << int(S) << endl;
    return int(S);
}

//---------------------------------------------------RANSAC optimal Homography Function----------------------------------------------------------------------
/**Method to find the optimal homography matrix using the RANSAC algorithm
* Preconditions: the vector correspondences is non-empty
* Postconditions: the method is able to identify the optimal homography using the number of inliers
* Assumptions: None
* @return a tuple of optimal number of inliners and homography matrix
*/
tuple<int, Mat> RANSAC_optimal(vector<pair<Mat, Mat>> correspondences, int num_rounds = -1) {
    Mat optimal_H;
    int optimal_inliers = 0;
    num_rounds = num_rounds > 0 ? num_rounds : rounds(0.15, 4, 0.95);
    for (int i = 0; i < num_rounds; i++) {
        // Random sample 4 keypoint pairs
        vector<pair<Mat, Mat>> copy = correspondences;

        // shuffle the copy using a random number generator
        random_device rd;
        mt19937 g(rd());
        shuffle(copy.begin(), copy.end(), g);
        vector<pair<Mat, Mat>> sample_corr(copy.begin(), copy.begin() + 4);

        // Compute the homography
        Mat H = homography(sample_corr);
        int num_inliers = 0;
        for (auto pair : correspondences) {

            Mat H = homography(sample_corr);
            Mat pt1 = pair.first;
            Mat pt2 = pair.second;

            // Project pt1 using H
            Mat pt1_homogeneous = (Mat_<double>(3, 1) << pt1.at<double>(0, 0), pt1.at<double>(0, 1), 1);
            Mat projected_pt1 = H * pt1_homogeneous;
            projected_pt1 /= projected_pt1.at<double>(2, 0);

            // Normalize pt2
            pt2 /= pt2.at<double>(0, 2);

            // Compute the loss
            double loss = norm(pt2 - projected_pt1.t());  // transpose projected_pt1 to match the layout of pt2

            if (loss <= 20) {
                num_inliers++;
            }
        }
        if (num_inliers > optimal_inliers) {
            optimal_H = H;
            optimal_inliers = num_inliers;
        }
    }
    return make_tuple(optimal_inliers, optimal_H);
}

//---------------------------------------------------------Visualize the Homography -------------------------------------------------------------------------
/** Method to visualize optimal homograpgy between the test image and query image (Work in Progress)
* Precondtions: H is a homography matrix between the input images
* Postconditions: i) The homography is visualized on the query image
                  ii) The overdrawn query image is saved into the directory
* Assumptions: The best match for the query image is already found
* @return: the visualized matrix result
*/
Mat visualize_homography(Mat img1, Mat img2, Mat H) {
    int h, w;
    h = img1.rows;
    w = img1.cols;

    // define the reference points
    vector<Point2f> pts;
    pts.push_back(Point2f(0, 0));
    pts.push_back(Point2f(0, h - 1));
    pts.push_back(Point2f(w - 1, h - 1));
    pts.push_back(Point2f(w - 1, 0));

    // transfer the points with affine transformation to get the new point on img2
    vector<Point2f> dst;
    perspectiveTransform(pts, dst, H);

    Mat result;
    img2.copyTo(result);
    polylines(result, dst, true, Scalar(0, 0, 255), 2, LINE_AA);
    imwrite("result.png", result);

    return result;
}
