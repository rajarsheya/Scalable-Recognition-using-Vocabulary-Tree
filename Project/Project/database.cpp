#include <iostream>
#include <filesystem>  // For directory traversal
#include <fstream>     // For file checking
#include <vector>
#include <cmath>
#include <map>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Structure to store feature vectors, kmeans cluster and related cluster information
struct VocabNode {
	// No OpenCV equivalent KMeans object in C++ API. OpenCV's k-means function — 
	// cv::kmeans — is procedural rather than object-oriented, meaning rather than creating 
	// a kmeans object and calling methods on it, we provide data to the function along with the 
	// number of clusters we want and other parameters, and it returns the centroid of the clusters it found.

	// Makes it a little more challenging to translate Python code to OpenCV in C+.
	// Since the kmeans object is being used to store the results of the k-means clustering, we can instead 
	// store the centroids and labels returned by the cv::kmeans function.

    Mat value; // Feature vector for this node
    Mat centers; // Centroids of clusters
    Mat labels; // Labels of each point
	vector<VocabNode*> children; // Child nodes
	map<string, int> occurrences_in_img;  // Mapping of image id to occurrences
	int index;  // Index of this node
};

class Database {

private:
    vector<Mat> images;  // to be defined
	string data_path;
    // map<string, pair<Mat, vector<KeyPoint>>> img_to_des_and_kpts;  // Assuming the image filename is used as the key
    int num_imgs;
    map<int, vector<string>> word_to_img;  // Assuming the word is an integer
    map<string, vector<float>> BoW;  // Assuming each word maps to a list of floats (histogram)
    vector<int> word_count;
    map<string, vector<float>> img_to_histogram;  // Assuming each image maps to a histogram
    vector<pair<Mat, string>> all_des;  // Assuming descriptor matrices are stored here
    vector<string> all_images;  // Assuming image paths are stored here
    vector<int> num_feature_per_image;
    vector<int> feature_start_idx;
    VocabNode* vocabulary_tree;
    int word_idx_count;

public:
    // Constructor
    Database() : 
		images{}, data_path{}, num_imgs{0}, word_to_img{}, BoW{}, word_count{}, 
        img_to_histogram{}, all_des{}, all_images{}, num_feature_per_image{},
        feature_start_idx{}, vocabulary_tree{nullptr}, word_idx_count{0} {
    }

    // void insert(Mat image) {  // Image class to be defined
    //     images.push_back(image);
    // }


	// uses Feature.cpp for FeatureDetector
	void Database::loadImgs(string data_path, string method = "SIFT") {
		this->data_path = data_path;

		// Assuming a FeatureDetector class exists that has a detect method
		FeatureDetector fd;

		for(auto &p : fs::recursive_directory_iterator(data_path)){
        	if(fs::is_regular_file(p)) {

				string img_path = p.path().string();
				Mat img = imread(img_path);

				// get all the keypoints and descriptors for each image
				vector<KeyPoint> kpts;
				Mat des;
				tie(kpts, des) = fd.detect(img, method); // Assuming that the detect method returns keypoints and descriptors

				// Append descriptors and image paths to all_des
				// Assuming that all_des is a vector of pairs
				for(int i = 0; i < des.rows; i++){
					Mat row = des.row(i);
					all_des.push_back(make_pair(row, img_path));
				}

				// Append image paths to all_image
				all_images.push_back(img_path);

				// Compute start index
				int idx = 0;
				if(!num_feature_per_image.empty())
					idx = num_feature_per_image.back() + feature_start_idx.back();

				// Append descriptor count to num_feature_per_image
				num_feature_per_image.push_back(des.rows);

				// Append start index to feature_start_idx
				feature_start_idx.push_back(idx);
			}
		}
				
		num_imgs = all_images.size();
	}


	void run_KMeans(int k, int L) {
		int total_nodes = (k * (pow(k, L)) - 1) / (k - 1);
		int n_leafs = pow(k, L);
		word_count = vector<int>(n_leafs, 0);  // Initialize all elements to zero

		vocabulary_tree = hierarchical_KMeans(k, L, all_des);
	}


	void print_tree(VocabNode* node) {
		vector<VocabNode*> children = node->children;
		if (children.size() == 0) {
			cout << node->index << endl;
		} else {
			for (VocabNode* c : children) {
				print_tree(c);
			}
		}
	}


	VocabNode* hierarchical_KMeans(int k, int L, vector<pair<Mat, string>>& des_and_path) {
		// Divide the given descriptor vector into k clusters
		vector<Mat> des;
		for (const auto& pair : des_and_path) {
			des.push_back(pair.first);
		}

		VocabNode* root = new VocabNode();
		int attempts = 5;
		TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.2);
		kmeans(des, k, root->labels, criteria, attempts, KMEANS_PP_CENTERS, root->centers);

		// If we reach the leaf node
		if (L == 0) {
			// Assign the index to the leaf nodes.
			root->index = word_idx_count++;
			// Count the number of occurrences of a word in an image used in tf-idf
			for (const auto& pair : des_and_path) {
				string img_path = pair.second;
				root->occurrences_in_img[img_path]++;
			}
			word_count[root->index] = root->occurrences_in_img.size();
			return root;
		}

		// If we are not on the leaf level, then for each cluster, 
		// we recursively run KMeans
		for (int i = 0; i < k; i++) {
			vector<pair<Mat, string>> cluster_i;
			for (int j = 0; j < des_and_path.size(); j++) {
				if (root->labels.at<int>(j) == i) {
					cluster_i.push_back(des_and_path[j]);
				}
			}
			VocabNode* node_i = hierarchical_KMeans(k, L - 1, cluster_i);
			root->children.push_back(node_i);
		}
		return root;
	}

	void build_histogram(VocabNode* node) {
		if (node->children.empty()) {
			for (auto const& [img, count] : node->occurrences_in_img) {
				if (img_to_histogram.find(img) == img_to_histogram.end()) {
					img_to_histogram[img] = vector<int>(word_idx_count, 0);
				}
				img_to_histogram[img][node->index] += count;
			}
		} else {
			for (auto child : node->children) {
				build_histogram(child);
			}
		}
	}

// Bag of words
	void Database::build_BoW() {
		for (size_t j = 0; j < all_image.size(); ++j) {
			std::string img = all_image[j];
			std::vector<double> t(word_idx_count, 0.0);
			t = img_to_histogram[img];
			for (int w = 0; w < word_idx_count; ++w) {
				double n_wj = img_to_histogram[img][w];
				double n_j = std::accumulate(img_to_histogram[img].begin(), img_to_histogram[img].end(), 0.0);
				double n_w = word_count[w];
				double N = num_imgs;
				t[w] = (n_wj/n_j) * std::log(N/n_w);
			}
			BoW[img] = t;
		}
	}

	// pings homography.cpp
	// vector<int> spatial_verification() {
	// 	// self, query, img_path_lst, method
	// 	return ['best_img', 'best_img_path', 'best_H'];
	// }

	VocabNode get_leaf_nodes() {
		// (self, root, des)
		return VocabNode();
	}

	// pings spatial_verification, which pings homography.cpp
    vector<int> query(Mat query_image, int k=10) {  // Image class to be defined
        vector<double> scores;
        for (const auto& db_image : images) {
            double score = db_image.compare(query_image);  // compare function to be defined in Image class
            scores.push_back(score);
        }

    //     // Sorting and extracting top-k scores in C++.
    //     // This will require additional code and possibly a custom comparator
    // }

	void save(const std::string &db_name) {
		std::ofstream file(db_name, std::ios::binary);

		// Assuming all member variables are public
		// Writing to the file
		// This might not work as expected with complex types
		file.write((char*)&data_path, sizeof(data_path));
		file.write((char*)&num_imgs, sizeof(num_imgs));
		file.write((char*)&word_to_img, sizeof(word_to_img));
		file.write((char*)&BoW, sizeof(BoW));
		file.write((char*)&word_count, sizeof(word_count));
		file.write((char*)&img_to_histgram, sizeof(img_to_histgram));
		file.write((char*)&all_des, sizeof(all_des));
		file.write((char*)&all_image, sizeof(all_image));
		file.write((char*)&num_feature_per_image, sizeof(num_feature_per_image));
		file.write((char*)&feature_start_idx, sizeof(feature_start_idx));
		file.write((char*)&word_idx_count, sizeof(word_idx_count));
		file.write((char*)&vocabulary_tree, sizeof(vocabulary_tree));

		file.close();
	}

	void load(const std::string &db_name) {
		std::ifstream file(db_name, std::ios::binary);

		// Assuming all member variables are public
		// Reading from the file
		// This might not work as expected with complex types
		file.read((char*)&data_path, sizeof(data_path));
		file.read((char*)&num_imgs, sizeof(num_imgs));
		file.read((char*)&word_to_img, sizeof(word_to_img));
		file.read((char*)&BoW, sizeof(BoW));
		file.read((char*)&word_count, sizeof(word_count));
		file.read((char*)&img_to_histgram, sizeof(img_to_histgram));
		file.read((char*)&all_des, sizeof(all_des));
		file.read((char*)&all_image, sizeof(all_image));
		file.read((char*)&num_feature_per_image, sizeof(num_feature_per_image));
		file.read((char*)&feature_start_idx, sizeof(feature_start_idx));
		file.read((char*)&word_idx_count, sizeof(word_idx_count));
		file.read((char*)&vocabulary_tree, sizeof(vocabulary_tree));

		file.close();
	}


};

void buildDatabase() {
	// params: load_path, k, L, method, save_path
    // print('Initial the Database')
    // db = Database()

    // print('Loading the images from {}, use {} for features'.format(load_path, method))
    // db.loadImgs(load_path, method=method)

    // print('Building Vocabulary Tree, with {} clusters, {} levels'.format(k, L))
    // db.run_KMeans(k=k, L=L)

    // print('Building Histgram for each images')
    // db.build_histgram(db.vocabulary_tree)

    // print('Building BoW for each images')
    // db.build_BoW()

    // print('Saving the database to {}'.format(save_path))
    // db.save(save_path)
}


int main(int argc, char* argv[]) {

	// Define the test path and DVD cover path
	string test_path = '../data/test';
	string cover_path = '../data/DVDcovers';
	// Initial and build the database
	Database db = Database();
	buildDatabase();
	// cover_path, k=5, L=5, method='SIFT', save_path='data_sift.txt'
	// If we have already build and save the database, we can just load database directly
	cout << 'Loading the database';
	db.load('data_sift.txt');

	// query a image
	Mat test = imread(test_path, '/image_01.jpeg');
	// best_img, best_img_path, best_H, best_K= db.query(test, top_K = 10, method='SIFT')

	return 0;
}
