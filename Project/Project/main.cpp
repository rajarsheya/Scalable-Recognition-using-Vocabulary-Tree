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


//----------------------------------------------Inverted File Vocabulary Tree Node Class---------------------------------------------------------------------

// Class to hold the attributes related to the vocabulary node/tf-idf (inverted file)
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

    /** Method to serialize the node to a FileStorage
    * Preconditions: i) FileStorage is already included in the program
    *                ii) The members of the node are initialized
    * Postconditions: The node is written into the file storage
    * Assumptions: None
    */
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

    /** Method to deserealize a file node into a Vocab Node
    *   Preconditions: The input FileNode is a valid node
    *   Postconditions: The attributes of the vocab node are instantiated
    *   Assumptions: None
    */
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

// Needed for FileStorage to work with VocabNode
void write(FileStorage& fs, const string&, const VocabNode& x) {
    x.write(fs);
}

// Needed for FileStorage to work with VocabNode
void read(const FileNode& node, VocabNode& x, const VocabNode& default_value = VocabNode()) {
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

// class to hold the attrbiutes related to the image path pair
class ImagePathPair {
public:
    Mat image;
    string path;
    // Default constructor
    ImagePathPair() {}
    ImagePathPair(const Mat& image, const string& path) : image(image), path(path) {}

    /** Method to write the pair into the filestorage
    * Preconditions: The image and path are non-empty
    * Postconditions: None
    */
    void write(FileStorage& fs) const {
        fs << "{";
        fs << "mat" << image;
        fs << "str" << path;
        fs << "}";
    }

    /** Method to deserealize file node contents into image path pairs
    * Preconditions: The input FileNode is valid
    * Postconditions: None
    */
    void read(const FileNode& node) {
        node["mat"] >> image;
        node["str"] >> path;
    }
};

// Needed for FileStorage to work with ImagePathPair
void write(FileStorage& fs, const string&, const ImagePathPair& x) {
    x.write(fs);
}

// Needed for FileStorage to work with ImagePathPair
void read(const FileNode& node, ImagePathPair& x, const ImagePathPair& default_value = ImagePathPair()) {
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

// Class to group toegether all attributes related to string vector pair
class StringVectorPair {
public:
    string str;
    vector<float> vec;
    // Default constructor
    StringVectorPair() {}
    // Constructor with parameters
    StringVectorPair(const std::string& str, const std::vector<float>& vec) : str(str), vec(vec) {}

    /** Method to write the pair into the filestorage
    * Preconditions: The string and vector are non-empty
    * Postconditions: None
    */
    void write(FileStorage& fs) const {
        fs << "{";
        fs << "str" << str;
        fs << "vec" << vec;
        fs << "}";
    }

    /** Method to deserealize file node contents into string vector pairs
    * Preconditions: The input FileNode is valid
    * Postconditions: None
    */
    void read(const FileNode& node) {
        node["str"] >> str;
        node["vec"] >> vec;
    }
};

// Needed for FileStorage to work with StringVectorPair
void write(cv::FileStorage& fs, const std::string&, const StringVectorPair& x) {
    x.write(fs);
}

// Needed for FileStorage to work with StringVectorPair
void read(const cv::FileNode& node, StringVectorPair& x, const StringVectorPair& default_value = StringVectorPair()) {
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}


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
int num_round_needed(double p, int k, double P) {
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

            if (loss <= 20){
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

/*
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
 */

 //--------------------------------------------------------------Database Class-------------------------------------------------------------------------------
  // class to group all the attrbitutes of the database construction
class Database {

private:
    string data_path;
    int num_imgs;
    map<int, vector<string>> word_to_img;  // Assuming the word is an integer
    map<string, vector<float>> BoW;  // Assuming each word maps to a list of floats (histogram)
    vector<int> word_count;
    map<string, vector<float>> img_to_histogram;  // Maps each image to a histogram
    vector<ImagePathPair> all_des;  //Desriptor for all images
    vector<string> all_images;  // Image paths
    map<string, Mat> frames; // Frames from video
    vector<int> num_feature_per_image;
    vector<int> feature_start_idx;
    VocabNode* vocabulary_tree;
    int word_idx_count;

public:
    //---------------------------------------------------------------Constructor-----------------------------------------------------------------------------
    /** Constructor method to initialize all the private members of the Database
    */
    Database() :
        data_path{}, num_imgs{ 0 }, word_to_img{}, BoW{}, word_count{},
        img_to_histogram{}, all_des{}, all_images{}, num_feature_per_image{},
        feature_start_idx{}, vocabulary_tree{ new VocabNode() }, word_idx_count{ 0 } {
    }

    //-------------------------------------------------------------Pre-Process the Images--------------------------------------------------------------------
    /** Method to perform pre-processing on the input image
    * Preconditions: i) Input mat is non-empty
    *                ii) fd is a valid feature detector
    * Postconditions: i) Each identifier descriptor is added to the overall descriptor list
    *                 ii) The image is added to the processed images list
    */
    void processImg(Mat img, string img_path, FeatureDetector1 fd, string method) {
        // get all the keypoints and descriptors for each image
        vector<KeyPoint> kpts;
        Mat des;
        tie(kpts, des) = fd.detect1(img, method);

        // Append descriptors and image paths to all_des
        for (int i = 0; i < des.rows; i++) {
            Mat row = des.row(i);
            all_des.push_back(ImagePathPair(row, img_path));
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
    /** Method to load all the images and/or videos in the selected directory using filesystem
    * Precondtions: The data_path is a readable directory
    * Postconditions: The total number of images is currently identified
    * Assumptions: Each video frame can be considered as a single image for every video in the directory
    */
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
                else {
                    // File is not a video. Check if the file has an image extension
                    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                        // Load the image
                        Mat img = imread(img_path);
                        processImg(img, img_path, fd, method);
                    }
                }
            }
        }

        num_imgs = (int)all_images.size();
        cout << "No. of images: " << num_imgs << endl;
    }

    //------------------------------------------------function for printing Vocab Tree-----------------------------------------------------------------------
    /** Method to print the Vocabulary tree
    * Precondtions: The vocabulary tree is correctly created using heirarchial k-means clustering
    * Postconditions: None
    */
    void print_tree(VocabNode* node) {
        vector<VocabNode*> children = node->children;
        if (children.size() == 0) {
            cout << node->index << " ";
        }
        else {
            cout << endl;
            for (VocabNode* c : children) {
                print_tree(c);
            }
        }
    }

    //---------------------------------------------------Hierarchical K-Means function-----------------------------------------------------------------------

    /** Method to construct heirarchial K-means tree recursively using bag of visual words
    * Precondtions: i) The descriptors are correctly matched to their respective paths
    *               ii) The branching factor k and the level L are well defined
    * Postconditions: The vocabulary tree groups the similar descruiptors together
    * Assumptions: None
    * @return the root node of the tree
    */
    VocabNode* hierarchical_KMeans(int k, int L, vector<ImagePathPair>& des_and_path) {
            // Divide the given descriptor vector into k clusters
        Mat descriptors;
        for (int i = 0; i < des_and_path.size(); i++) {
            descriptors.push_back(des_and_path[i].image);
        }
        descriptors.convertTo(descriptors, CV_32F);
        VocabNode* root = new VocabNode();
        Mat labels, centers;
        int attempts = 10;
        TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01);

        // If we reach the leaf node
        if (L == 0) {
            // Assign the index to the leaf nodes.
            root->index = word_idx_count++;
            // Count the number of occurrences of a word in an image used in tf-idf
            for (const auto& pair : des_and_path) {
                string img_path = pair.path;
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

                // If we are not on the leaf level, then for each cluster,
                // we recursively run KMeans
                for (int i = 0; i < k; i++) {
                    vector<ImagePathPair> cluster_i;
                    for (int j = 0; j < des_and_path.size(); j++) {
                        if (root->labels.total() > 0 && root->labels.at<int>(j) == i) {
                            cluster_i.push_back(des_and_path[j]);
                        }
                    }
                    if (!cluster_i.empty() && root->labels.total() > 0) {
                        try {
                            VocabNode* node_i = hierarchical_KMeans(k, L - 1, cluster_i);
                            root->children.push_back(node_i);
                        }
                        catch (const cv::Exception& e) {
                            cerr << "Caught OpenCV exception in hierarchical_KMeans: " << e.what() << endl;
                            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
                        }
                    }

                }

            }
            else {
                // Work in Progress
                // Adjust the number of clusters or skip the kmeans() call. // WORK ON THIS
                //cout<< "The number of descriptors is less than K. Choose a lower value for K in K-means" << endl;
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
                    //CHANGE CODE
                    // vector<ImagePathPair> cluster_i;
                    for (int j = 0; j < des_and_path.size(); j++) {
                        if (root->labels.total() > 0 && root->labels.at<int>(j) == i) {
                            des_and_path[j].first.convertTo(des_and_path[j].first, CV_32F);
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
                            VocabNode* node_i = hierarchical_KMeans(k, L - 1, cluster_i);
                            root->children.push_back(node_i);
                        } catch (const cv::Exception& e) {
                            cerr << "Caught OpenCV exception in hierarchical_KMeans: " << e.what() << endl;
                            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
                        }
                    }
                }
                */
            }
        }
        catch (const cv::Exception& e) {
            cerr << "Caught OpenCV exception in kmeans: " << e.what() << endl;
            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
        }
        return root;
    }

    //--------------------------------------------------------------Build Histogram--------------------------------------------------------------------------
    /** Method to build histograms for the leaf nodes
    * Precondtions: The input node pointer is valid for the given vocab tree
    * Postconditions: None
    * Assumptions: The nodes above the leaf nodes are virtual nodes
    */
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
    /** Method to build the bag od visual words for all the images in the tree
    * Precondtions: The images are correctly converted into their histograms
    * Postconditions: The bag of visual words for each image holds the identified histograms
    * Assumptions: None
    */
    void build_BoW() {
        for (size_t j = 0; j < all_images.size(); ++j) {
            string img = all_images[j];
            vector<float> t(word_idx_count, 0.0);
            t = img_to_histogram[img];
            for (int w = 0; w < word_idx_count; ++w) {
                float n_wj = img_to_histogram[img][w];
                float n_j = accumulate(img_to_histogram[img].begin(), img_to_histogram[img].end(), 0.0);
                float n_w = word_count[w];
                float N = all_images.size();
                t[w] = (n_wj / n_j) * log(N / n_w);
            }
            BoW[img] = t;
        }
    }

    //-----------------------------------------------------Spatial Verification------------------------------------------------------------------------------
    /** Method to perform spatial vertification for the query image against a list of image paths
    * Preconditons: i) The input Mat query is non-empty
    *               ii) The input vector contains valid paths to the images
    * Postconditions: None
    * @return a tuple containing best matching image, its path and the homography matrix
    */
    vector<pair<string, int>> spatial_verification(Mat& query, vector<string>& img_path_list, string& method) {
        FeatureDetector1 fd;
      //int best_inliers = numeric_limits<int>::lowest();
        vector<string> best_img_path;
        vector<Mat> best_img, best_H;
        vector<pair<string, int> >fileinlier;

        for (const string& img_path : img_path_list) {
            Mat img;
            // Check if the best match is a frame from a video
            if (img_path.find("_frame_") != string::npos) {
                // The best match is a frame from a video, retrieve it from the frames map
                img = frames[img_path];
            }
            else {
                // The best match is not a frame from a video, load the image from the path
                img = imread(img_path);
            }

            auto correspondences = fd.detectAndMatch(img, query, method);

            int inliers;
            Mat optimal_H;
            tie(inliers, optimal_H) = RANSAC_find_optimal_Homography(correspondences, 10);
            cout << "Running RANSAC... Image: " << img_path << " Inliers: " << inliers << endl;

            fileinlier.push_back(make_pair(img_path, inliers));


        }
        return fileinlier;
    }

    //----------------------------------------------------function to get the leaf nodes---------------------------------------------------------------------
    /** Method to extract the leaf nodes of the constructed vocabulary tree
    * Preconditons: i) The input node is a valid node in the tree
    *               ii) The vector of descriptors is non-empty
    * Postconditions:None
    * @return the pointer to the closest child node
    */
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
    /** Method to find the best matches for the query image using the vocabulary tree
    * Preconditons: i) The Mat query is non-empty
    *               ii) The method is a valid feature detector
    * Postconditions: The best k matches for the query image are found
    * Assumptions: Every helper functions are correctly created and compiled
    * @return a tuple containing the best image, its path, the homography matrix and top k matches
    */
    tuple<vector<string>, vector<string>> query(Mat input_img, int top_K, string method) {
        FeatureDetector1 fd;
        vector<KeyPoint> kpts;
        Mat des;

        // compute the features
        tie(kpts, des) = fd.detect1(input_img, method);

        cout << "word_idx_count = " << word_idx_count << endl;
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
            float N = all_images.size();
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
                if (find(target_img_lst.begin(), target_img_lst.end(), img) == target_img_lst.end()) {
                    target_img_lst.push_back(img);
                }
            }
        }

        // compute similarity between query BoW and the all targets
        vector<pair<string, double>> score_lst(target_img_lst.size());
        for (size_t j = 0; j < target_img_lst.size(); ++j) {
            string img = target_img_lst[j];
            vector<float> t = BoW[img];
            //score calculationg using L1-norm
            score_lst[j].first = img;
            score_lst[j].second = 2 + accumulate(begin(q), end(q), 0.0f) - accumulate(begin(t), end(t), 0.0f);
        }
        cout << "score_lst size: " << score_lst.size() << endl;
        double average_score = 0;
        for (int i = 0; i < score_lst.size(); i++) {
            cout << "i=" << i << " score[i] file=" << score_lst[i].first << " score[i]=" << score_lst[i].second << endl;
            average_score = average_score + score_lst[i].second;
        }
        average_score = average_score / score_lst.size();
        cout << "Average Score: " << average_score << endl;

        // sort the similarity and take the top_K most similar image
        // get top_K best match images
        vector<int> indices(score_lst.size());
        iota(indices.begin(), indices.end(), 0); // Filling the indices vector with values from 0 to size-1

        sort(indices.begin(), indices.end(),
            [&score_lst](int i1, int i2) { return score_lst[i1] < score_lst[i2]; }); // Sort indices based on corresponding scores

        int actual_top_K = static_cast<int>(indices.size());

        vector<int> best_K_match_imgs_idx(indices.end() - actual_top_K, indices.end());
        reverse(best_K_match_imgs_idx.begin(), best_K_match_imgs_idx.end());

        vector<string> best_K_match_imgs(actual_top_K);
        transform(best_K_match_imgs_idx.begin(), best_K_match_imgs_idx.end(), best_K_match_imgs.begin(),
            [&target_img_lst](int i) { return target_img_lst[i]; });

        vector<Mat> best_img;
        vector<string> best_img_path;
      
        //vector<pair<Mat, Mat>> best_correspondences;
        vector<pair<string, int> >fileinlier1;

        fileinlier1 = spatial_verification(input_img, best_K_match_imgs, method);

        vector<pair<double, string>> Final_score_lst(target_img_lst.size());
        for (int i = 0; i < fileinlier1.size(); i++) {
            cout << fileinlier1[i].first << " " << fileinlier1[i].second << " " << score_lst[i].first << " " << score_lst[i].second << endl;
            for (int j = 0; j < fileinlier1.size(); j++) {
                if (fileinlier1[i].first == score_lst[j].first) {
                    double new_score = score_lst[j].second * (double)fileinlier1[i].second;
                    Final_score_lst[i].first = new_score;
                    Final_score_lst[i].second = score_lst[j].first;
                    continue;
                }
            }
        }
        sort(Final_score_lst.begin(), Final_score_lst.end());
        reverse(Final_score_lst.begin(), Final_score_lst.end());
        cout << "Final score_lst size: " << Final_score_lst.size() << endl;
        double final_average_score = 0;
        for (int i = 0; i < Final_score_lst.size(); i++) {
            cout << "i=" << i << " Final_score_lst[i] file=" << Final_score_lst[i].second << " Final_score_lst[i]=" << Final_score_lst[i].first << endl;
            final_average_score = final_average_score + Final_score_lst[i].first;
        }
        final_average_score = final_average_score / Final_score_lst.size();
        cout << "Final Average Score: " << final_average_score << endl;

        for (int i = 0; i < top_K; i++) {
            Mat img_temp = imread(Final_score_lst[i].second);
            best_img_path.push_back(Final_score_lst[i].second);
        }
        fd.drawCircle(input_img, kpts);
        //visualize_homography(input_img, best_img, best_H);
       
        //visualize_homography(input_img, best_img, best_H, best_correspondences);

        return make_tuple(best_img_path, best_K_match_imgs);
    }

    //-----------------------------------------------function for Running K-meanns algorithm-----------------------------------------------------------------
    /** Method to run the K-means clustering algroithm
    * Preconditions: i) The branching factor and level are already defined
    * Postconditions: The number of nodes and leaf nodes are calculated
    * Assumptions: None
    */
    void run_KMeans(int k, int L) {
        int total_nodes = (k * (pow(k, L)) - 1) / (k - 1);
        cout << "Total Nodes = " << total_nodes << endl;
        int n_leafs = pow(k, L);
        cout << "Total Leaf Nodes = " << n_leafs << endl;
        word_count = vector<int>(n_leafs, 0);  // Initialize all elements to zero
        try {
            vocabulary_tree = hierarchical_KMeans(k, L, all_des);
        }
        catch (const cv::Exception& e) {
            cerr << "Caught OpenCV exception: " << e.what() << endl;
            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
        }
    }

    //--------------------------------------------------------------Saving the Database----------------------------------------------------------------------
    /** Method to save the constructed vocabulary tree into the designated directory
    * Preconditons:i) The input db_name is a valid file
    *              ii) The vocabulary tee is contructed
    * Postconditions: Tehe database is saved into a desginated file
    * Assumptions: None
    */
    void save(const string& db_name) {
        FileStorage fs(db_name, FileStorage::WRITE);
        fs << "data_path" << data_path;
        fs << "word_count" << word_count;
        fs << "word_idx_count" << word_idx_count;
        // Uses ImagePathPair
        fs << "all_des" << all_des;
        // Vector objects
        fs << "num_feature_per_image" << num_feature_per_image;
        fs << "feature_start_idx" << feature_start_idx;
        fs << "all_images" << all_images;
        // Convert maps to vectors
        // map<string, Mat> frames; 
        vector<StringVectorPair> bowVec;
        for (const auto& pair : BoW) {
            bowVec.push_back(StringVectorPair(pair.first, pair.second));
        }
        fs << "BoW" << bowVec;
        vector<StringVectorPair> imgHistVec;
        for (const auto& pair : img_to_histogram) {
            imgHistVec.push_back(StringVectorPair(pair.first, pair.second));
        }
        fs << "img_to_histogram" << imgHistVec;
        // Uses VocabNode
        fs << "Vocab_Tree" << *vocabulary_tree;
        fs.release();
    }


    //--------------------------------------------------------------Loading the Database---------------------------------------------------------------------
    /** Method to load the vocabulary tree from the saved file
     * Preconditions:i) The input filename is valid
     *               ii) The vocabulary tree is correctly stored
     * Posconditions: The file is closed after reading
     * Assumptions: None
     */
    void load(const string& db_name) {
        FileStorage fs(db_name, FileStorage::READ);
        fs["data_path"] >> data_path;
        fs["word_count"] >> word_count;

        fs["word_idx_count"] >> word_idx_count;
        // Uses ImagePathPair
        fs["all_des"] >> all_des;
        // For vector objects
        fs["num_feature_per_image"] >> num_feature_per_image;
        fs["feature_start_idx"] >> feature_start_idx;
        fs["all_images"] >> all_images;
        // Convert vector back to map
        // map<string, Mat> frames; 
        vector<StringVectorPair> bowVec;
        fs["BoW"] >> bowVec;
        BoW.clear();
        for (const auto& pair : bowVec) {
            BoW[pair.str] = pair.vec;
        }
        vector<StringVectorPair> imgHistVec;
        fs["img_to_histogram"] >> imgHistVec;
        img_to_histogram.clear();
        for (const auto& pair : imgHistVec) {
            img_to_histogram[pair.str] = pair.vec;
        }
        // For user-defined types
        fs["Vocab_Tree"] >> *vocabulary_tree;
        fs.release();
    }


    //--------------------------------------------------------------Building the Database--------------------------------------------------------------------
    /** Method to build the database using the specified branching factor, level and feature detector
     * Preconditions: i) The input load_path holds the database images
     *                ii) The branching factor and level is already defined
     *                iii) The method name is a valid feature detector
     * Postconditions: None
     * Assumptions: The helper functions are correctly created and compiled
     */
    void buildDatabase(string load_path, int k, int L, string method) {
        cout << "Loading the images from " << load_path << ", use " << method << " for features\n";
        loadImgs(load_path, method);

        cout << "Building Vocabulary Tree, with " << k << " branching factor,and " << L << " levels\n";

        try {
            run_KMeans(k, L);
        }
        catch (const cv::Exception& e) {
            cerr << "Caught OpenCV exception: " << e.what() << endl;
            cerr << "Error occurred at k = " << k << ", L = " << L << endl;
        }

        cout << "Building Histogram for each images\n";
        build_histogram(vocabulary_tree);

      //cout << "Vocab_tree = " << endl;
      // print_tree(vocabulary_tree);

        cout << "Building BoW for each images\n";
        build_BoW();
    }

};

//----------------------------------------------------------Driver Code - main Class-------------------------------------------------------------------------
/** Driver method to read the images, build the vocabulary tree and display the similar images
* Preconditions: i) The test path and cover path are valid strings
*                ii) The cover path directory and the test image are non-empty
* Postconditions: i) The best match image gets displayed to the screen
*                 ii) The time taken and other specifications are printed to the console
*                 iii) A key is preseed to terminate the session
* Assumtpions: i) The helper functions are correctly created and compiled
*              ii) The OpenCV module and the required header files are propoerly configured
*/
int main(int argc, char* argv[]) {
    //Define the query image path and Image Dataset path
    string test_path = "./data/query";
    string cover_path = "./data/DVDCovers";

    string fdname;
    int fdnumber;
    cout << "Enter the feature detector number from the following: " << endl;
    cout << "1 - SIFT (Recommended)" << endl;
    cout << "2 - ORB" << endl;
    cout << "3 - BRISK" << endl;
    cout << "4 - AKAZE" << endl;
    cin >> fdnumber;
    if (fdnumber == 1) {
        fdname = "SIFT";
    }
    else if (fdnumber == 2) {
        fdname = "ORB";
    }
    else if (fdnumber == 3) {
        fdname = "BRISK";
    }
    else if (fdnumber == 4) {
        fdname = "AKAZE";
    }
    else {
        cout << "Enter Valid Number for the feature detector!" << endl;
    }

    // The counter for the number of folders
    int folder_count = 0;

    // Create a recursive directory iterator
    recursive_directory_iterator dir_iter(cover_path);

    // Loop over the directory entries
    for (const auto& entry : dir_iter)
    {
        // Check if the entry is a directory
        if (is_directory(entry))
        {
            // Increment the folder count
            folder_count++;
        }
    }

    // Print the number of folders
    cout << "There are " << folder_count << " folders in " << cover_path << endl;

    string img_path = test_path + "/query_01.jpg";

    Mat test = imread(img_path);
    vector<string> best_img_path;
    vector<cv::String> best_K;

    //if(folder_count == 0){
        // Initial and build the database
    Database db;
    
    // Build database
    cout << "Building the database...\n";
    std::chrono::time_point<std::chrono::high_resolution_clock> startdbbuild, enddbbuild;
    startdbbuild = std::chrono::high_resolution_clock::now();
    db.buildDatabase(cover_path, 3, 5, fdname);
    enddbbuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_db_build = enddbbuild - startdbbuild;
    cout << "Database Built\n";
    cout << "Time taken to build the database: " << Time_for_db_build.count() << " sec" << endl;

    // Save the database
    cout << "Saving the database...\n";
    std::chrono::time_point<std::chrono::high_resolution_clock> startdbsave, enddbsave;
    startdbsave = std::chrono::high_resolution_clock::now();
    db.save("DVD_DB_50.txt");
    enddbsave = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_db_save = enddbsave - startdbsave;
    cout << "Database saved\n";
    cout << "Time taken to save the database: " << Time_for_db_save.count() << " sec" << endl;
    
    //Uncomment the load function call when you want to load an already existing database and comment the build and save segments
    /*
    // Load the database
    cout << "Loading the database...\n";
    std::chrono::time_point<std::chrono::high_resolution_clock> startdbload, enddbload;
    startdbload = std::chrono::high_resolution_clock::now();
    db.load("DVD_DB_50.txt");
    enddbload = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_db_load = enddbload - startdbload;
    cout << "Database loaded\n";
    cout << "Time taken to load the database: " << Time_for_db_load.count() << " sec" << endl;
    */

    // Query an image
    cout << "Querying the image...\n";
    std::chrono::time_point<std::chrono::high_resolution_clock> startquery, endquery;
    startquery = std::chrono::high_resolution_clock::now();
    tie(best_img_path, best_K) = db.query(test, 3, fdname);
    endquery = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > Time_for_querying = endquery - startquery;
    cout << "Querying the image done!\n";
    cout << "Time taken for querying: " << Time_for_querying.count() << " sec" << endl;

    cout << "best_img_path = " << best_img_path[0] << endl;
    for (int i = 0; i < best_img_path.size(); i++) {
        cout << "i=" << i << "best_img_path = " << best_img_path[i] << endl;
    }
    //}
    /*
    else{
        for(int i=0;i<folder_count;i++){
            string index = to_string(i+1);
            string folder_cover_path = cover_path + "/p" + index;
            cout << "\nFolder path=" << folder_cover_path << endl;
            // Initial and build the database
            Database db;

            // Build database
            cout << "Building the database...\n";
            std::chrono::time_point<std::chrono::high_resolution_clock> startdbbuild, enddbbuild;
            startdbbuild = std::chrono::high_resolution_clock::now();
            db.buildDatabase(folder_cover_path, 3, 5, fdname);
            enddbbuild = std::chrono::high_resolution_clock::now();
            std::chrono::duration< double > Time_for_db_build = enddbbuild - startdbbuild;
            cout << "Database Built\n";
            cout << "Time taken to build the database: " << Time_for_db_build.count() << " sec" << endl;

            // Save the database
            cout << "Saving the database...\n";
            std::chrono::time_point<std::chrono::high_resolution_clock> startdbsave, enddbsave;
            startdbsave = std::chrono::high_resolution_clock::now();
            db.save("./Database_" + folder_cover_path + ".txt");
            enddbsave = std::chrono::high_resolution_clock::now();
            std::chrono::duration< double > Time_for_db_save = enddbsave - startdbsave;
            cout << "Database saved\n";
            cout << "Time taken to save the database: " << Time_for_db_save.count() << " sec" << endl;

            //Uncomment the load function call when you want to load an already existing database
            /*
             // Load the database
             cout << "Loading the database...\n";
             std::chrono::time_point<std::chrono::high_resolution_clock> startdbload, enddbload;
             startdbload = std::chrono::high_resolution_clock::now();
             db.load("Database_" + cover_path + ".txt");
             enddbload = std::chrono::high_resolution_clock::now();
             std::chrono::duration< double > Time_for_db_load = enddbload - startdbload;
             cout << "Database loaded\n";
             cout << "Time taken to load the database: " << Time_for_db_load.count() << " sec" << endl;
             *

            // Query an image
            cout << "Querying the image...\n";

            std::chrono::time_point<std::chrono::high_resolution_clock> startquery, endquery;
            startquery = std::chrono::high_resolution_clock::now();
            tie(best_img_path, best_K) = db.query(test, 5, fdname);
            endquery = std::chrono::high_resolution_clock::now();
            std::chrono::duration< double > Time_for_querying = endquery - startquery;
            cout << "Querying the image done!\n";
            cout << "Time taken for querying: " << Time_for_querying.count() << " sec" << endl;

            cout << "best_img_path = " << best_img_path[0] << endl;
            for(int i=0;i<best_img_path.size();i++){
                cout << "i=" << i << "best_img_path = " << best_img_path[i] << endl;
            }
        }
    }*/

    // Display the test image
    namedWindow("Test Image", WINDOW_NORMAL);
    imshow("Test Image", test);


    for (int i = 0; i < best_img_path.size(); i++) {
        Mat best_img = imread(best_img_path[i]);
        cout << "i=" << i << endl;
        // Display the best matching images
        namedWindow("Best Match", WINDOW_NORMAL);
        imshow("Best Match", best_img);

        waitKey(0);
    }
    return 0;
}
