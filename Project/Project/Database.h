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

#include "FeatureDetector1.h"
#include "VocabNode.h"
#include "Homography.h"

using namespace std;
using namespace cv;
//mespace fs = __fs::filesystem; // for MacOS
namespace fs = std::experimental::filesystem; // for Windows
using namespace fs;

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
            tie(inliers, optimal_H) = RANSAC_optimal(correspondences, 10);
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
            vector<float>descriptor;
            des.row(i).copyTo(descriptor);
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
        // Load best match image
        Mat best_img = imread(best_img_path[0]);

        // Uncomment the visualize_homography() call below to see the homography visualization-------------------------------
        // visualize_homography(input_img, best_img, method);

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