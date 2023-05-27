#include <opencv2/opencv.hpp>
#include <tuple>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;
using namespace cv;

Mat homography(vector<pair<Mat, Mat>> correspondences) {
    int n = correspondences.size(); // number of points
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

/*
    find the number of rounds needed for RANSAC to have P chance to success.
    Inputs:
    - p: the probability that the sample match is inlier
    - k: number of matches we need for one round
    - P: the probability success after S rounds
 */
int num_round_needed(double p, int k, double P)
{
    double S = log(1 - P) / log(1 - pow(p, k));
    return int(S);
}


// Find the optimal homography matrix using RANSAC algorithm
// Input: a vector of pairs that store the correspondences
// Output: a 3x3 homography matrix
// Return a tuple of int and cv::Mat
tuple<int, Mat> RANSAC_find_optimal_Homography(vector<pair<Mat, Mat>> correspondences, int num_rounds = -1)
{
    Mat optimal_H;
    int optimal_inliers = 0;
    num_rounds = num_rounds > 0 ? num_rounds : num_round_needed(0.15, 4, 0.95);
    for (int i = 0; i < num_rounds; i++)
    {
        // Random sample 4 keypoint pairs
        //vector<pair<Mat, Mat>> sample_corr;
        //sample(correspondences.begin(), correspondences.end(), back_inserter(sample_corr), 4, mt19937{ random_device{}() });
        vector<pair<Mat, Mat>> copy = correspondences;

        // shuffle the copy using a random number generator
        random_device rd;
        mt19937 g(rd());
        random_shuffle(copy.begin(), copy.end(), g);
        vector<pair<Mat, Mat>> sample_corr(copy.begin(), copy.begin() + 4);
        // Compute the homography
        Mat H = homography(sample_corr);
        int num_inliers = 0;
        for (auto pair : correspondences)
        {
            Mat pt1 = pair.first;
            Mat pt2 = pair.second;
            // Project pt1 using H
            Mat projected_pt1 = H * pt1;
            projected_pt1 /= projected_pt1.at<double>(2, 0);
            double loss = cv::norm(pt2 - projected_pt1);
            if (loss <= 20)
            {
                num_inliers++;
            }
        }
        if (num_inliers > optimal_inliers)
        {
            optimal_H = H;
            optimal_inliers = num_inliers;
        }
    }
    return make_tuple(optimal_inliers, optimal_H);
}

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