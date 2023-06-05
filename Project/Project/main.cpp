//-----------------------------------------------------------------------------------------------------------------------------------------------------------
//
//  Enhanced Vocabulary Trees for Real-Time Object Recognition in Image and Video Streams
//  Team members:  Arsheya Raj, Sugam Jaiswal, Josiah Zacharias
//
//  This project is inspired on the research paper which has the topic as "Scalable Recognition with a Vocabulary Tree".
//  Paper Link: https://ieeexplore-ieee-org.offcampus.lib.washington.edu/document/1641018
//
//-----------------------------------------------------------------------------------------------------------------------------------------------------------

#include <tuple>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <ctime>
#include <time.h>   // this is needed for high resolution clock
#include <chrono>   // for high resolution clock
//#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
//#include <experimental/filesystem>

using namespace std;
using namespace cv;
namespace fs = std::__fs::filesystem;

//-------------------------------------------------Inverted File Vocabulary Tree Node------------------------------------------------------------------------
// Store feature vectors, kmeans cluster and related cluster information
class VocabNode {
    // No OpenCV equivalent KMeans object in C++ API. OpenCV's k-means function —
    // cv::kmeans — is procedural rather than object-oriented, meaning rather than creating
    // a kmeans object and calling methods on it, we provide data to the function along with the
    // number of clusters we want and other parameters, and it returns the centroid of the clusters it found.
    public:
    Mat value; // Feature vector for this node
    Mat centers; // Centroids of clusters
    Mat labels; // Labels of each point
    vector<VocabNode*> children; // Child nodes
    map<string, int> occurrences_in_img;  // Mapping of image id to occurrences
    int index;  // Index of this node

    // Serialize this node to a FileStorage
    void write(FileStorage& fs) const {
        fs << "{";
        fs << "value" << value;
        fs << "centers" << centers;
        fs << "labels" << labels;
        fs << "index" << index;
        fs << "occurrences_in_img" << "[";
        for (const auto& pair : occurrences_in_img) {
            fs << "{:" << "image_id" << pair.first << "occurrences" << pair.second << "}";
        }
        fs << "]";
        fs << "children" << "[";
        for (const auto& child : children) {
            child->write(fs);
        }
        fs << "]";
        fs << "}";
    }

    // Deserialize a node from a FileNode
    void read(const FileNode& node) {
        node["value"] >> value;
        node["centers"] >> centers;
        node["labels"] >> labels;
        node["index"] >> index;
        FileNode occurrencesNode = node["occurrences_in_img"];
        for (FileNodeIterator it = occurrencesNode.begin(); it != occurrencesNode.end(); ++it) {
            string image_id;
            int occurrences;
            (*it)["image_id"] >> image_id;
            (*it)["occurrences"] >> occurrences;
            occurrences_in_img[image_id] = occurrences;
        }
        FileNode childrenNode = node["children"];
        for (FileNodeIterator it = childrenNode.begin(); it != childrenNode.end(); ++it) {
            VocabNode* child = new VocabNode();
            child->read(*it);
            children.push_back(child);
        }
    }
};

static void write(FileStorage& fs, const string&, const VocabNode& x) {
    x.write(fs);
}

static void read(const FileNode& node, VocabNode& x, const VocabNode& default_value = VocabNode()) {
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

static void write(FileStorage& fs, const string&, const pair<Mat, string>& x) {
    fs << "{";
    fs << "Mat" << x.first;
    fs << "String" << x.second;
    fs << "}";
}

static void read(const FileNode& node, pair<Mat, string>& x, const pair<Mat, string>& default_value = pair<Mat, string>()) {
    if(node.empty())
        x = default_value;
    else {
        node["Mat"] >> x.first;
        node["String"] >> x.second;
    }
}

//--------------------------------------------------------Feature Detector Class-----------------------------------------------------------------------------
class FeatureDetector1 {
private:
    Ptr<FeatureDetector> sift = SIFT::create(); //SIFT
    Ptr<FeatureDetector> orb = ORB::create();   //ORB
    Ptr<FeatureDetector> brisk = BRISK::create();   //BRISK
    Ptr<FeatureDetector> akaze = AKAZE::create();   //AKAZE

public:
    //--------------------------------------------------------Keypoint Detection-----------------------------------------------------------------------------
    tuple<vector<KeyPoint>, Mat> detect1(Mat& img1, string method) {
        Mat gray1;
        cvtColor(img1, gray1, COLOR_BGR2GRAY);
        vector<KeyPoint> kp1;
        Mat des1;
        if (method == "SIFT") {
            sift->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = SIFT::create();
            extractor->compute(gray1, kp1, des1);
            //sift->detectAndCompute(gray1, noArray(), kp1, des1);
        }
        else if (method == "ORB") {
            orb->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = ORB::create();
            extractor->compute(gray1, kp1, des1);
            //orb->detectAndCompute(gray1, noArray(), kp1, des1);
        }
        else if (method == "BRISK") {
            brisk->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = BRISK::create();
            extractor->compute(gray1, kp1, des1);
            //orb->detectAndCompute(gray1, noArray(), kp1, des1);
        }
        else if (method == "AKAZE") {
            akaze->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = AKAZE::create();
            extractor->compute(gray1, kp1, des1);
            //orb->detectAndCompute(gray1, noArray(), kp1, des1);
        }
        return make_tuple(kp1,des1);
    }
    
    //--------------------------------------------------Matching Function for Keypoints----------------------------------------------------------------------
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
    vector<pair<Mat, Mat>> detectAndMatch(Mat& img1, Mat& img2, string method) {
        Mat gray1, gray2;
        cvtColor(img1, gray1, COLOR_BGR2GRAY);
        cvtColor(img2, gray2, COLOR_BGR2GRAY);

        vector<KeyPoint> kp1, kp2;
        Mat des1, des2;

        if (method == "SIFT") {
            sift->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = SIFT::create();
            extractor->compute(gray1, kp1, des1);
            sift->detect(gray2, kp2);
            Ptr<DescriptorExtractor> extractor1 = SIFT::create();
            extractor1->compute(gray2, kp2, des2);
            //sift->detectAndCompute(gray1, noArray(), kp1, des1);
            //sift->detectAndCompute(gray2, noArray(), kp2, des2);
        }
        else if (method == "ORB") {
            orb->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = ORB::create();
            extractor->compute(gray1, kp1, des1);
            orb->detect(gray2, kp2);
            Ptr<DescriptorExtractor> extractor1 = ORB::create();
            extractor1->compute(gray2, kp2, des2);
            //orb->detectAndCompute(gray1, noArray(), kp1, des1);
            //orb->detectAndCompute(gray2, noArray(), kp2, des2);
        }
        else if (method == "BRISK") {
            brisk->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = BRISK::create();
            extractor->compute(gray1, kp1, des1);
            brisk->detect(gray2, kp2);
            Ptr<DescriptorExtractor> extractor1 = BRISK::create();
            extractor1->compute(gray2, kp2, des2);
            //brisk->detectAndCompute(gray1, noArray(), kp1, des1);
            //brisk->detectAndCompute(gray2, noArray(), kp2, des2);
        }
        else if (method == "AKAZE") {
            akaze->detect(gray1, kp1);
            Ptr<DescriptorExtractor> extractor = AKAZE::create();
            extractor->compute(gray1, kp1, des1);
            akaze->detect(gray2, kp2);
            Ptr<DescriptorExtractor> extractor1 = AKAZE::create();
            extractor1->compute(gray2, kp2, des2);
            //akaze->detectAndCompute(gray1, noArray(), kp1, des1);
            //akaze->detectAndCompute(gray2, noArray(), kp2, des2);
        }

        // Create a matcher
        vector<DMatch> matches;
        // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
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

//-----------------------------------------------------------Homography Function-----------------------------------------------------------------------------
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
/*
// Inputs:
//    - p: the probability that the sample match is inlier
//    - k: number of matches we need for one round
//    - P: the probability success after S rounds
*/
int num_round_needed(double p, int k, double P) {
    double S = log(1 - P) / log(1 - pow(p, k));
    return int(S);
}

//---------------------------------------------------RANSAC optimal Homography Function----------------------------------------------------------------------
/*
// Find the optimal homography matrix using RANSAC algorithm
// Input: a vector of pairs that store the correspondences
// Output: a 3x3 homography matrix
// Returns a tuple of int and cv::Mat
*/
tuple<int, Mat> RANSAC_find_optimal_Homography(vector<pair<Mat, Mat>> correspondences, int num_rounds = -1) {
    Mat optimal_H;
    int optimal_inliers = 0;
    num_rounds = num_rounds > 0 ? num_rounds : num_round_needed(0.15, 4, 0.95);
    for (int i = 0; i < num_rounds; i++) {
        // Random sample 4 keypoint pairs
        vector<pair<Mat, Mat>> copy = correspondences;

        // shuffle the copy using a random number generator
        random_device rd;
        mt19937 g(rd());
        shuffle(copy.begin(), copy.end(),g);
        vector<pair<Mat, Mat>> sample_corr(copy.begin(), copy.begin() + 4);
        
        // Compute the homography
        Mat H = homography(sample_corr);
        int num_inliers = 0;
        for (auto pair : correspondences) {
            
            Mat H = homography(sample_corr);
            Mat pt1 = pair.first;
            Mat pt2 = pair.second;
            
            // Project pt1 using H
            Mat pt1_homogeneous = (Mat_<double>(3,1) << pt1.at<double>(0, 0), pt1.at<double>(0, 1), 1);
            Mat projected_pt1 = H * pt1_homogeneous;
            projected_pt1 /= projected_pt1.at<double>(2, 0);
            
            // Normalize pt2
            pt2 /= pt2.at<double>(0, 2);
            
            // Compute the loss
            double loss = cv::norm(pt2 - projected_pt1.t());  // transpose projected_pt1 to match the layout of pt2
            
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

void visualize_homography(Mat& img1, Mat& img2, Mat& H, vector<pair<Mat, Mat>>& correspondences) {
    // Create a new image that can contain both images side by side
    int max_height = max(img1.rows, img2.rows);
    int total_width = img1.cols + img2.cols;
    Mat result(max_height, total_width, img1.type(), Scalar(0, 0, 0));

    // Copy img1 and img2 into result
    Mat roi1(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(roi1);
    Mat roi2(result, Rect(img1.cols, 0, img2.cols, img2.rows));
    img2.copyTo(roi2);

    // Convert the homography matrix to float
    Mat H_float;
    H.convertTo(H_float, CV_32F);

    // Apply the homography to the corners of the first image
    vector<Point2f> corners1(4);
    corners1[0] = Point2f(0, 0);
    corners1[1] = Point2f(img1.cols, 0);
    corners1[2] = Point2f(img1.cols, img1.rows);
    corners1[3] = Point2f(0, img1.rows);
    vector<Point2f> corners2(4);
    perspectiveTransform(corners1, corners2, H_float);

    // Offset the points in the second image by the width of the first image
    for (Point2f& pt : corners2) {
        pt.x += img1.cols;
    }

    // Draw the transformed corners as a quadrilateral
    vector<Point> corners2_int(corners2.begin(), corners2.end());
    polylines(result, corners2_int, true, Scalar(0, 0, 255), 2, LINE_AA);

    // Draw lines between the corresponding points
    for (const auto& correspondence : correspondences) {
        Point2f pt1(correspondence.first.at<double>(0, 0), correspondence.first.at<double>(0, 1));
        Point2f pt2(correspondence.second.at<double>(0, 0), correspondence.second.at<double>(0, 1));

        line(result, pt1, Point(pt2.x + img1.cols, pt2.y), Scalar(0, 255, 0), 1);
    }

    // Display the visualization
    namedWindow("Homography", WINDOW_NORMAL);
    imshow("Homography", result);
    waitKey(0);
}


// void visualize_homography(Mat img1, Mat img2, Mat H) {
//     int h, w;
//     h = img1.rows;
//     w = img1.cols;

//     // define the reference points
//     vector<Point2f> pts;
//     pts.push_back(Point2f(0, 0));
//     pts.push_back(Point2f(0, h - 1));
//     pts.push_back(Point2f(w - 1, h - 1));
//     pts.push_back(Point2f(w - 1, 0));

//     // transfer the points with affine transformation to get the new point on img2
//     vector<Point2f> dst;
//     perspectiveTransform(pts, dst, H);

//     Mat result;
//     img2.copyTo(result);
//     vector<Point> dst_int;
//     for (const auto& point : dst) {
//         dst_int.push_back(Point(cvRound(point.x), cvRound(point.y)));
//     }
//     polylines(result, dst_int, true, Scalar(0, 0, 255), 2, LINE_AA);
//     // polylines(result, dst, true, Scalar(0, 0, 255), 2, LINE_AA);
//     imwrite("result.png", result);
// }

//--------------------------------------------------------------Database Class-------------------------------------------------------------------------------
class Database {

private:
    string data_path;
    int num_imgs;
    map<int, vector<string>> word_to_img;  // Assuming the word is an integer
    map<string, vector<float>> BoW;  // Assuming each word maps to a list of floats (histogram)
    vector<int> word_count;
    map<string, vector<float>> img_to_histogram;  // Maps each image to a histogram
    vector<pair<Mat, string>> all_des;  // Descriptor matrices
    vector<string> all_images;  // Image paths
    map<string, Mat> frames; // Frames from video
    vector<int> num_feature_per_image;
    vector<int> feature_start_idx;
    VocabNode* vocabulary_tree;
    int word_idx_count;

public:
    //---------------------------------------------------------------Constructor-----------------------------------------------------------------------------
    Database() :
        data_path{}, num_imgs{ 0 }, word_to_img{}, BoW{}, word_count{},
        img_to_histogram{}, all_des{}, all_images{}, num_feature_per_image{},
        feature_start_idx{}, vocabulary_tree{ nullptr }, word_idx_count{ 0 } {
    }
    
    //-------------------------------------------------------------Pre-Process the Images--------------------------------------------------------------------
    void processImg(Mat img, string img_path, FeatureDetector1 fd, string method) {
            // get all the keypoints and descriptors for each image
            vector<KeyPoint> kpts;
            Mat des;
            tie(kpts, des) = fd.detect1(img, method);

            // Append descriptors and image paths to all_des
            for (int i = 0; i < des.rows; i++) {
                Mat row = des.row(i);
                all_des.push_back(make_pair(row, img_path));
            }

            // Append image paths to all_image
            all_images.push_back(img_path);

            // Compute start index
            int idx = 0;
            if (!num_feature_per_image.empty())
                idx = num_feature_per_image.back() + feature_start_idx.back();

            // Append descriptor count to num_feature_per_image
            num_feature_per_image.push_back(des.rows);

            // Append start index to feature_start_idx
            feature_start_idx.push_back(idx);
    }
    
    //-------------------------------------------------------Load Images in databse function-----------------------------------------------------------------
    void loadImgs(string data_path, string method) {
        this->data_path = data_path;

        // Assuming a FeatureDetector1 class exists that has a detect method
        FeatureDetector1 fd;

        for (auto& p : fs::recursive_directory_iterator(data_path)) {
            if (fs::is_regular_file(p)) {
                string img_path = p.path().string();
                string extension = p.path().extension().string();
                transform(extension.begin(), extension.end(), extension.begin(), ::tolower); // Convert the extension to lower case

                if (extension == ".avi" || extension == ".mp4" || extension == ".mkv" || extension == ".flv"
                                        || extension == ".mov" || extension == ".wmv") {
                                    // File is a video
                    cout << "Found video: " << img_path << ". Processing frames..." << endl;
                    VideoCapture cap(img_path);
                    Mat frame;
                    int frameNumber = 0;
                    while (cap.read(frame)) {
                        // Process frames
                        string frame_path = img_path + "_frame_" + to_string(frameNumber);
                        frames[frame_path] = frame.clone();
                        processImg(frame, frame_path, fd, method);
                        frameNumber++;
                    }
                }
                else{
                    // File is not a video. Check if the file has an image extension
                    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                        // Load the image
                        Mat img = imread(img_path);
                        processImg(img, img_path, fd, method);
                    }
                }
                
//
//                // Check if the file has an image extension
//                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
//                    // Load the image
//                    Mat img = imread(img_path);
//
//                    // get all the keypoints and descriptors for each image
//                    vector<KeyPoint> kpts;
//                    Mat des;
//                    tie(kpts, des) = fd.detect1(img, method);
//
//                    // Append descriptors and image paths to all_des
//                    for (int i = 0; i < des.rows; i++) {
//                        Mat row = des.row(i);
//                        all_des.push_back(make_pair(row, img_path));
//                    }
//
//                    // Append image paths to all_image
//                    all_images.push_back(img_path);
//
//                    // Compute start index
//                    int idx = 0;
//                    if (!num_feature_per_image.empty())
//                        idx = num_feature_per_image.back() + feature_start_idx.back();
//
//                    // Append descriptor count to num_feature_per_image
//                    num_feature_per_image.push_back(des.rows);
//
//                    // Append start index to feature_start_idx
//                    feature_start_idx.push_back(idx);
//
//                } else { // Not an image file. Skip it.
//                    continue;
//                }
            }
        }

        num_imgs = (int)all_images.size();
        cout << "No. of images: " << num_imgs << endl;
    }

    //------------------------------------------------function for printing Vocab Tree-----------------------------------------------------------------------
    void print_tree(VocabNode* node) {
        vector<VocabNode*> children = node->children;
        if (children.size() == 0) {
            cout << node->index << endl;
        }
        else {
            cout << node->index << " -> ";
            for (VocabNode* c : children) {
                print_tree(c);
            }
        }
    }

    //---------------------------------------------------Hierarchical K-Means function-----------------------------------------------------------------------
    VocabNode* hierarchical_KMeans(int k, int L, vector<pair<Mat, string>>& des_and_path) {
        // Divide the given descriptor vector into k clusters
        Mat descriptors;
        for (int i = 0; i < des_and_path.size(); i++) {
            descriptors.push_back(des_and_path[i].first);
        }
        descriptors.convertTo(descriptors, CV_32F);
        VocabNode* root = new VocabNode();
        Mat labels, centers;
        int attempts = 5;
        TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.00000001);
        
        // If we reach the leaf node (base case)
        if (L == 0) {
            // Assign the index to the leaf nodes.
            root->index = word_idx_count++;
            // Count the number of occurrences of a word in an image used in tf-idf
            for (const auto& pair : des_and_path) {
                string img_path = pair.second;
                root->occurrences_in_img[img_path]++;
            }
            word_count[root->index] = (int)root->occurrences_in_img.size();
            return root;
        }
        
        try {
            if (descriptors.rows >= k) {
                kmeans(descriptors, k, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);
                root->labels = labels;
                root->centers = centers;
                //cout<< "centers:" << centers << endl;
                
                // If we are not on the leaf level, then for each cluster,
                // we recursively run KMeans
                for (int i = 0; i < k; i++) {
                    vector<pair<Mat, string>> cluster_i;
                    for (int j = 0; j < des_and_path.size(); j++) {
                        if (root->labels.total() > 0 && root->labels.at<int>(j) == i) {
                            cluster_i.push_back(des_and_path[j]);
                        }
                    }
                    if (!cluster_i.empty() && root->labels.total() > 0) {
                        try {
                        VocabNode* node_i = hierarchical_KMeans(k, L - 1, cluster_i);
                        root->children.push_back(node_i);
                        } catch (const cv::Exception& e) {
                            cerr << "Caught OpenCV exception in hierarchical_KMeans: " << e.what() << endl;
                            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
                        }
                    }
                    
                }
                
            } else {
                // Adjust the number of clusters or skip the kmeans() call. // WORK ON THIS
                cout<< "The number of descriptors is less than K. Choose a lower value for K in K-means" << endl;
                /*
                int rows = k - descriptors.rows;
                kmeans(descriptors, descriptors.rows, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);
                for (int i = 0; i < rows; i++) {
                    root->labels = labels;
                    root->centers = centers;
                }
                // If we are not on the leaf level, then for each cluster,
                // we recursively run KMeans
                for (int i = 0; i < k; i++) {
                    vector<pair<Mat, string>> cluster_i;
                    for (int j = 0; j < des_and_path.size(); j++) {
                        if (root->labels.total() > 0 && root->labels.at<int>(j) == i) {
                            cluster_i.push_back(des_and_path[j]);
                        }
                    }
                    if (!cluster_i.empty() && root->labels.total() > 0) {
                        try {
                            VocabNode* node_i = hierarchical_KMeans(k, L - 1, cluster_i);
                            root->children.push_back(node_i);
                        } catch (const cv::Exception& e) {
                            cerr << "Caught OpenCV exception in hierarchical_KMeans: " << e.what() << endl;
                            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
                        }
                    }
                    else{
                        try {
                            VocabNode* node_i = hierarchical_KMeans(k, L - 1, des_and_path);
                            root->children.push_back(node_i);
                        } catch (const cv::Exception& e) {
                            cerr << "Caught OpenCV exception in hierarchical_KMeans: " << e.what() << endl;
                            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
                        }
                    }
                }
                */
            }
        } catch (const cv::Exception& e) {
            cerr << "Caught OpenCV exception in kmeans: " << e.what() << endl;
            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
        }

        return root;
    }

    //--------------------------------------------------------------Build Histogram--------------------------------------------------------------------------
    void build_histogram(VocabNode* node) {
        // Build the histgram for the leaf nodes
        if (node->children.empty()) {
            for (auto const& occ : node->occurrences_in_img) {
                if (img_to_histogram.find(occ.first) == img_to_histogram.end()) {
                    img_to_histogram[occ.first] = vector<float>(word_idx_count, 0);
                }
                img_to_histogram[occ.first][node->index] += occ.second;
            }
        }
        else {
            for (auto child : node->children) {
                build_histogram(child);
            }
        }
    }

    //--------------------------------------------------------------Bag of Words-----------------------------------------------------------------------------
    void build_BoW() {
        for (size_t j = 0; j < all_images.size(); ++j) {
            string img = all_images[j];
            vector<float> t(word_idx_count, 0.0);
            t = img_to_histogram[img];
            for (int w = 0; w < word_idx_count; ++w) {
                float n_wj = img_to_histogram[img][w];
                float n_j = accumulate(img_to_histogram[img].begin(), img_to_histogram[img].end(), 0.0);
                float n_w = word_count[w];
                float N = num_imgs;
                t[w] = (n_wj / n_j) * log(N / n_w);
            }
            BoW[img] = t;
        }
    }

    //-----------------------------------------------------Spatial Verification------------------------------------------------------------------------------
    tuple<Mat, string, Mat, vector<pair<Mat, Mat>>> spatial_verification(Mat& query, vector<string>& img_path_list, string& method) {
        FeatureDetector1 fd;
        int best_inliers = numeric_limits<int>::lowest();
        string best_img_path;
        Mat best_img, best_H;
        vector<pair<Mat, Mat>> best_correspondences;

        for (const string& img_path : img_path_list) {
            Mat img;
            // Check if the best match is a frame from a video
            if (img_path.find("_frame_") != string::npos) {
                // The best match is a frame from a video, retrieve it from the frames map
                img = frames[img_path];
            }
            else{
                // The best match is not a frame from a video, load the image from the path
                img = imread(img_path);
            }
            
            auto correspondences = fd.detectAndMatch(img, query, method);
            
            int inliers;
            Mat optimal_H;
            tie(inliers, optimal_H) = RANSAC_find_optimal_Homography(correspondences, 2000);

            cout << "Running RANSAC... Image: " << img_path << " Inliers: " << inliers << endl;

            if (best_inliers < inliers) {
                best_inliers = inliers;
                best_img_path = img_path;
                best_img = img;
                best_H = optimal_H;
                best_correspondences = correspondences;
            }
        }
        return make_tuple(best_img, best_img_path, best_H, best_correspondences);
    }

    //----------------------------------------------------function to get the leaf nodes---------------------------------------------------------------------
    VocabNode* get_leaf_nodes(VocabNode* node, vector<float>& descriptor) {
        // If this node has no children, it is a leaf node.
        if (node->children.empty()) {
            return node;
        }

        // Find the child node whose center is closest to the descriptor.
        VocabNode* closest_child = node->children[0];
        float closest_distance = norm(descriptor, node->centers.row(0), NORM_L2);

        for (int i = 1; i < node->children.size(); ++i) {
            float distance = norm(descriptor, node->centers.row(i), NORM_L2);
            if (distance < closest_distance) {
                closest_child = node->children[i];
                closest_distance = distance;
            }
        }
        
        // Recurse on the closest child node.
        return get_leaf_nodes(closest_child, descriptor);
    }

    //--------------------------------------------------------------query image function---------------------------------------------------------------------
    tuple<Mat, string, Mat, vector<string>> query(Mat input_img, int top_K, string method) {
        FeatureDetector1 fd;
        vector<KeyPoint> kpts;
        Mat des;
        
        //fd.drawCircle(input_img, kpts);
        // compute the features
        tie(kpts, des) = fd.detect1(input_img, method);
        
        cout << "word_idx_count = " << word_idx_count << endl;
        //word_idx_count = 10000;
        vector<float> q(word_idx_count, 0.0);
        vector<VocabNode*> node_lst;

        // Assuming des is a Mat with a row for each descriptor
        for (int i = 0; i < des.rows; i++) {
            Mat row = des.row(i);
            vector<float> descriptor(row.begin<float>(), row.end<float>());
            VocabNode* node = get_leaf_nodes(vocabulary_tree, descriptor);
            node_lst.push_back(node);
            q[node->index] += 1;
        }

        for (int w = 0; w < word_idx_count; ++w) {
            float n_w = word_count[w];
            float N = num_imgs;
            float n_wq = q[w];
            float n_q = accumulate(begin(q), end(q), 0.0f);
            q[w] = (n_wq / n_q) * log(N / n_w);
        }

        // get a list of img from database that have the same visual words
        vector<string> target_img_lst;
        for (auto n : node_lst) {
            if (n == nullptr) continue; // Skip virtual nodes
            for (auto const& entry : n->occurrences_in_img) {
                string img = entry.first;
                int count = entry.second;
                //cout << "path of image = " << img << endl;
                //cout<< "Number of img from database that have the same visual words: " << count << endl;
                if (find(target_img_lst.begin(), target_img_lst.end(), img) == target_img_lst.end()) {
                    target_img_lst.push_back(img);
                }
            }
        }

        // compute similarity between query BoW and the all targets
        vector<double> score_lst(target_img_lst.size(), 0.0);
        for (size_t j = 0; j < target_img_lst.size(); ++j) {
            string img = target_img_lst[j];
            vector<float> t = BoW[img];
            // lower scores mean closer match between images
            score_lst[j] = 2 + accumulate(begin(q), end(q), 0.0f) - accumulate(begin(t), end(t), 0.0f);
        }
        
        // sort the similarity and take the top_K most similar image
        // get top_K best match images
        vector<int> indices(score_lst.size());
        iota(indices.begin(), indices.end(), 0); // Filling the indices vector with values from 0 to size-1

        sort(indices.begin(), indices.end(),
            [&score_lst](int i1, int i2) { return score_lst[i1] < score_lst[i2]; }); // Sort indices based on corresponding scores

        int actual_top_K = min(top_K, static_cast<int>(indices.size()));

        vector<int> best_K_match_imgs_idx(indices.end() - actual_top_K, indices.end());
        reverse(best_K_match_imgs_idx.begin(), best_K_match_imgs_idx.end());

        vector<string> best_K_match_imgs(actual_top_K);
        transform(best_K_match_imgs_idx.begin(), best_K_match_imgs_idx.end(), best_K_match_imgs.begin(),
            [&target_img_lst](int i) { return target_img_lst[i]; });

        Mat best_img;
        string best_img_path;
        Mat best_H;
        vector<pair<Mat, Mat>> best_correspondences;
        tie(best_img, best_img_path, best_H, best_correspondences) = spatial_verification(input_img, best_K_match_imgs, method);
        
        cout << "best_img_path_q = " << best_img_path << endl;
        
        //fd.drawCircle(input_img, kpts);
        //fd.drawCircle(best_img, best_H);
        visualize_homography(input_img, best_img, best_H, best_correspondences);
        
        return make_tuple(best_img, best_img_path, best_H, best_K_match_imgs);
    }
    
    //-----------------------------------------------function for Running K-meanns algorithm-----------------------------------------------------------------
    void run_KMeans(int k, int L) {
        int total_nodes = (k * (pow(k, L)) - 1) / (k - 1);
        cout << "Total Nodes = " << total_nodes << endl;
        int n_leafs = pow(k, L);
        cout << "Total Leaf Nodes = " << n_leafs << endl;
        word_count = vector<int>(n_leafs, 0);  // Initialize all elements to zero
        try {
//            for(int i=0;i<all_des.size();i++){
//                cout<< "i=" <<i<<endl;
//                cout<< all_des[i].first <<endl;
//                cout<< all_des[i].second <<endl;
//            }
            vocabulary_tree = hierarchical_KMeans(k, L, all_des);
        } catch (const cv::Exception& e) {
            cerr << "Caught OpenCV exception: " << e.what() << endl;
            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
        }
    }

    //--------------------------------------------------------------Saving the Database----------------------------------------------------------------------
    void save(const string& db_name) {
        FileStorage fs(db_name, FileStorage::WRITE);

        fs << "data_path" << data_path;
        fs << "num_imgs" << num_imgs;
        fs << "word_count" << word_count;
        fs << "word_idx_count" << word_idx_count;

        // For cv::Mat objects
        fs << "all_des" << all_des;

        // For std::vector objects
        fs << "num_feature_per_image" << num_feature_per_image;
        fs << "feature_start_idx" << feature_start_idx;

        // For std::map objects
        fs << "word_to_img" << "{";
        for (const auto& pair : word_to_img) {
            fs << pair.first << pair.second;
        }
        fs << "}";

        fs << "BoW" << "{";
        for (const auto& pair : BoW) {
            fs << pair.first << pair.second;
        }
        fs << "}";

        fs << "img_to_histogram" << "{";
        for (const auto& pair : img_to_histogram) {
            fs << pair.first << pair.second;
        }
        fs << "}";

        // For user-defined types
        fs << "Vocab Tree" << *vocabulary_tree;

        fs.release();
    }

    //--------------------------------------------------------------Loading the Database---------------------------------------------------------------------
    void load(const string& db_name) {
        FileStorage fs(db_name, FileStorage::READ);

        fs["data_path"] >> data_path;
        fs["num_imgs"] >> num_imgs;
        fs["word_count"] >> word_count;
        fs["word_idx_count"] >> word_idx_count;

        // For cv::Mat objects
        fs["all_des"] >> all_des;

        // For std::vector objects
        fs["num_feature_per_image"] >> num_feature_per_image;
        fs["feature_start_idx"] >> feature_start_idx;

        // For std::map objects
        FileNode word_to_img_node = fs["word_to_img"];
        for (FileNodeIterator it = word_to_img_node.begin(); it != word_to_img_node.end(); ++it) {
            int key;
            vector<string> value;
            it >> key >> value;
            word_to_img[key] = value;
        }

        FileNode BoW_node = fs["BoW"];
        for (FileNodeIterator it = BoW_node.begin(); it != BoW_node.end(); ++it) {
            string key;
            vector<float> value;
            it >> key >> value;
            BoW[key] = value;
        }

        FileNode img_to_histogram_node = fs["img_to_histogram"];
        for (FileNodeIterator it = img_to_histogram_node.begin(); it != img_to_histogram_node.end(); ++it) {
            string key;
            vector<float> value;
            it >> key >> value;
            img_to_histogram[key] = value;
        }

        // For user-defined types
        fs["Vocab Tree"] >> *vocabulary_tree;

        fs.release();
    }

    //--------------------------------------------------------------Building the Database--------------------------------------------------------------------
    void buildDatabase(string load_path, int k, int L, string method, string save_path) {
        cout << "Loading the images from " << load_path << ", use " << method << " for features\n";
        loadImgs(load_path, method);

        cout << "Building Vocabulary Tree, with " << k << " clusters, " << L << " levels\n";

        try {
            run_KMeans(k, L);
        } catch (const cv::Exception& e) {
            cerr << "Caught OpenCV exception: " << e.what() << endl;
            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
        }

        cout << "Building Histogram for each images\n";
        build_histogram(vocabulary_tree);
        
//        cout << "Vocab_tree = " << endl;
//        print_tree(vocabulary_tree);

        cout << "Building BoW for each images\n";
        build_BoW();

        cout << "Saving the database to " << save_path << "\n";
        save(save_path);
    }
};

//void mserExtractor (const Mat& image){
//    Ptr<MSER> ms = MSER::create();
//    vector<vector<Point> > regions;
//    vector<cv::Rect> mser_bbox;
//    ms->detectRegions(image, regions, mser_bbox);
//
//    for (int i = 0; i < regions.size(); i++){
//        rectangle(image, mser_bbox[i], CV_RGB(0, 255, 0));
//    }
//
//    imshow("mser", image);
//    waitKey(0);
//}

//----------------------------------------------------------Driver Code - main Class-------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    //Define the query image path and Image Dataset path
    // string test_path = "./data/video";
    string test_path = "./data/DVD-DB-50";
    // string query_path = "./data/query";
    string query_path = "./data/DVD-Query-50-1";

    string fdname;
    int fdnumber =1;
    // cout << "Enter the feature detector number from the following: " << endl;
    // cout << "1 - SIFT (Recommended)" << endl;
    // cout << "2 - ORB" << endl;
    // cout << "3 - BRISK" << endl;
    // cout << "4 - AKAZE" << endl;
    // cin >> fdnumber;
    if(fdnumber == 1){
        fdname = "SIFT";
    }
    else if(fdnumber == 2){
        fdname = "ORB";
    }
    else if(fdnumber == 3){
        fdname = "BRISK";
    }
    else if(fdnumber == 4){
        fdname = "AKAZE";
    }
    
    // Initial and build the database
    Database db;

    // Build database
    cout << "Building the database...\n";
    std::chrono::time_point<std::chrono::high_resolution_clock> startdbbuild, enddbbuild;
    startdbbuild = std::chrono::high_resolution_clock::now();
    db.buildDatabase(test_path, 5, 5, fdname, "Database_DVD_DB_50.txt");
    enddbbuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_db_build = enddbbuild - startdbbuild;
    cout << "Database Built\n";
    cout << "Time taken to build the database: " << Time_for_db_build.count() << " sec" << endl;
    
    /*
    // Save the database
    cout << "Saving the database...\n";
    std::chrono::time_point<std::chrono::high_resolution_clock> startdbsave, enddbsave;
    startdbsave = std::chrono::high_resolution_clock::now();
    db.save("Database_DVD_DB_50.txt");
    enddbsave = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_db_save = enddbsave - startdbsave;
    cout << "Database saved\n";
    cout << "Time taken to save the database: " << Time_for_db_save.count() << " sec" << endl;
    */
    
    /*
    // Load the database
    cout << "Loading the database...\n";
    std::chrono::time_point<std::chrono::high_resolution_clock> startdbload, enddbload;
    startdbload = std::chrono::high_resolution_clock::now();
    db.load("Database_DVD_DB_50.txt");
    enddbload = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_db_load = enddbload - startdbload;
    cout << "Database loaded\n";
    cout << "Time taken to load the database: " << Time_for_db_load.count() << " sec" << endl;
    */
    
    // Query an image
    string query = "/16.jpg";
    string img_path = query_path + query;
    cout << "Querying image " << query;
    Mat test = imread(img_path);
    Mat best_img;
    string best_img_path;
    Mat best_H;
    vector<cv::String> best_K;
    std::chrono::time_point<std::chrono::high_resolution_clock> startquery, endquery;
    startquery = std::chrono::high_resolution_clock::now();
    tie(best_img, best_img_path, best_H, best_K) = db.query(test, 5, fdname);
    endquery = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_querying = endquery - startquery;
    cout << "Querying the image done!\n";
    cout << "Time taken for querying: " << Time_for_querying.count() << " sec" << endl;
    
    cout << "best_img_path = " << best_img_path << endl;

    //mserExtractor(test);
    
    // Display the test image
    namedWindow("Test Image", WINDOW_NORMAL);
    imshow("Test Image", test);

    // Display the best matching image
    namedWindow("Best Match", WINDOW_NORMAL);
    imshow("Best Match", best_img);

    waitKey(0);
    
    return 0;
}
