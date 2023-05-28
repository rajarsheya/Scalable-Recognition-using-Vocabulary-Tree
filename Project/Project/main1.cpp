#include <opencv2/opencv.hpp>
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
#include <experimental/filesystem>

using namespace std;
using namespace cv;

//feature.cpp
class FeatureDetector {
private:
    Ptr<SIFT> sift = SIFT::create();
    Ptr<ORB> orb = ORB::create();

public:

    void detect(Mat& img1, vector<KeyPoint>& kp1, Mat& des1, string method = "SIFT") {
        Mat gray1;
        cvtColor(img1, gray1, COLOR_BGR2GRAY);

        if (method == "SIFT") {
            sift->detectAndCompute(gray1, noArray(), kp1, des1);
        }
        else if (method == "ORB") {
            orb->detectAndCompute(gray1, noArray(), kp1, des1);
        }
    }

    vector<tuple<Point2f, Point2f, float>> match(
        vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
        Mat& des1, Mat& des2) {
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

    vector<tuple<Point2f, Point2f, float>> detectAndMatch(
        Mat& img1, Mat& img2, string method = "SIFT") {
        Mat gray1, gray2;
        cvtColor(img1, gray1, COLOR_BGR2GRAY);
        cvtColor(img2, gray2, COLOR_BGR2GRAY);

        vector<KeyPoint> kp1, kp2;
        Mat des1, des2;

        if (method == "SIFT") {
            sift->detectAndCompute(gray1, noArray(), kp1, des1);
            sift->detectAndCompute(gray2, noArray(), kp2, des2);
        }
        else if (method == "ORB") {
            orb->detectAndCompute(gray1, noArray(), kp1, des1);
            orb->detectAndCompute(gray2, noArray(), kp2, des2);
        }

        return match(kp1, kp2, des1, des2);
    }

    void drawCircle(Mat& image, vector<KeyPoint>& kp) {
        int H = image.rows;
        int W = image.cols;

        for (int i = 0; i < kp.size(); i++) {
            Point2f pt = kp[i].pt;
            circle(image, Point(pt.x, pt.y), static_cast<int>(kp[i].size), Scalar(0, 255, 0), 1);
        }
    }
};

    vector<tuple<Point2f, Point2f, float>> filterMatchPoints(vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
        Mat& des1, Mat& des2) {

        vector<tuple<Point2f, Point2f, float>> result;

        for (int i = 0; i < des1.rows; i++) {
            float smallestDistance = numeric_limits<float>::max();
            float secondSmallestDistance = numeric_limits<float>::max();
            int smallestIdx = 0;

            for (int j = 0; j < des2.rows; j++) {
                float distance = norm(des1.row(i), des2.row(j), NORM_L2);

                if (distance < smallestDistance) {
                    secondSmallestDistance = smallestDistance;
                    smallestDistance = distance;
                    smallestIdx = j;
                }
            }

            float ratio = smallestDistance / secondSmallestDistance;

            if (ratio < 0.8) {
                Point2f pt1 = kp1[i].pt;
                Point2f pt2 = kp2[smallestIdx].pt;
                result.push_back(std::make_tuple(pt2, pt1, ratio));
            }
        }

        return result;
    }

    vector<tuple<Point2f, Point2f, float>> SIFTMatchPoints(Mat& img1, Mat& img2) {
        Mat gray1, gray2;
        cvtColor(img1, gray1, COLOR_BGR2GRAY);
        cvtColor(img2, gray2, COLOR_BGR2GRAY);
        Ptr<cv::SIFT> sift = SIFT::create();

        vector<KeyPoint> kp1, kp2;
        Mat des1, des2;

        sift->detectAndCompute(gray1, noArray(), kp1, des1);
        sift->detectAndCompute(gray2, noArray(), kp2, des2);

        return filterMatchPoints(kp1, kp2, des1, des2);
    }

    vector<KeyPoint> SIFTMatchPointsSingle(Mat& img1) {
        Mat gray1;
        cvtColor(img1, gray1,COLOR_BGR2GRAY);
        Ptr<SIFT> sift = SIFT::create();

        vector<KeyPoint> kp1;
        Mat des1;

        sift->detectAndCompute(gray1, noArray(), kp1, des1);

        return kp1;
    }

    Mat draw_circle(Mat& image, vector<KeyPoint>& kp) {
        int H = image.rows;
        int W = image.cols;

        Mat result = image.clone();

        for (int i = 0; i < kp.size(); i++) {
            Point2f pt = kp[i].pt;
            circle(result, Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                static_cast<int>(kp[i].size), Scalar(0, 255, 0), 1);
        }

        return result;
    }

// Homograhy.cpp
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

//database.cpp
namespace fs = std::experimental::filesystem;

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
			images{}, data_path{}, num_imgs{ 0 }, word_to_img{}, BoW{}, word_count{},
			img_to_histogram{}, all_des{}, all_images{}, num_feature_per_image{},
			feature_start_idx{}, vocabulary_tree{ nullptr }, word_idx_count{ 0 } {
		}


		// uses Feature.cpp for FeatureDetector
		// NEED TO INTEGRATE WITH OTHER CPP FILES
		void loadImgs(string data_path, string method = "SIFT") {
			this->data_path = data_path;

			// Assuming a FeatureDetector class exists that has a detect method
			FeatureDetector fd;

			for (auto& p : fs::recursive_directory_iterator(data_path)) {
				if (fs::is_regular_file(p)) {

					string img_path = p.path().string();
					Mat img = imread(img_path);

					// get all the keypoints and descriptors for each image
					vector<KeyPoint> kpts;
					Mat des;
					// ---------------------------- NEXT LINE NEEDS TO BE INTEGRATED WITH FEATURE.CPP --------------------------------
					tie(kpts, des) = fd.detect(img, method); // Assuming that the detect method returns keypoints and descriptors

					// Append descriptors and image paths to all_des
					// Assuming that all_des is a vector of pairs
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
			}
			else {
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
			// Build the histgram for the leaf nodes
			if (node->children.empty()) {
				for (auto const& [img, count] : node->occurrences_in_img) {
					if (img_to_histogram.find(img) == img_to_histogram.end()) {
						img_to_histogram[img] = vector<float>(word_idx_count, 0);
					}
					img_to_histogram[img][node->index] += count;
				}
			}
			else {
				for (auto child : node->children) {
					build_histogram(child);
				}
			}
		}


		// Bag of words
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


		// pings homography.cpp
		// NEED TO INTEGRATE WITH OTHER CPP FILES
		tuple<Mat, string, Mat> spatial_verification(const Mat& query, const vector<string>& img_path_list, const string& method) {
			FeatureDetecter fd;
			int best_inliers = numeric_limits<int>::lowest();
			string best_img_path;
			Mat best_img, best_H;

			for (const string& img_path : img_path_list) {
				Mat img = imread(img_path);
				// ---------------------------- NEXT LINEs NEEDS TO BE INTEGRATED WITH FEATURE.CPP & HOMOGRAPHY.CPP --------------------------------
				auto correspondences = fd.detect_and_match(img, query, method); // Assuming detect_and_match returns a suitable data structure
				auto [inliers, optimal_H] = RANSAC_find_optimal_Homography(correspondences, 2000); // Assuming RANSAC_find_optimal_Homography returns a pair

				cout << "Running RANSAC... Image: " << img_path << " Inliers: " << inliers << endl;

				if (best_inliers < inliers) {
					best_inliers = inliers;
					best_img_path = img_path;
					best_img = img;
					best_H = optimal_H;
				}
			}
			return make_tuple(best_img, best_img_path, best_H);
		}


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


		// pings spatial_verification, which pings homography.cpp
		tuple<Mat, string, Mat, vector<string>> query(Mat input_img, int top_K, string method) {
			FeatureDetecter fd;
			vector<KeyPoint> kpts;
			Mat des;

			// compute the features
			tie(kpts, des) = fd.detect(input_img, method); // TODO: replace with actual C++ version when available

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
				for (auto const& [img, count] : n->occurrences_in_img) {
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
				score_lst[j] = 2 + accumulate(begin(q), end(q), 0.0f) - accumulate(begin(t), end(t), 0.0f);
			}

			// sort the similarity and take the top_K most similar image
			// get top_K best match images
			vector<int> indices(score_lst.size());
			iota(indices.begin(), indices.end(), 0); // Filling the indices vector with values from 0 to size-1

			sort(indices.begin(), indices.end(),
				[&score_lst](int i1, int i2) { return score_lst[i1] < score_lst[i2]; }); // Sort indices based on corresponding scores

			vector<int> best_K_match_imgs_idx(indices.end() - top_K, indices.end());
			reverse(best_K_match_imgs_idx.begin(), best_K_match_imgs_idx.end());

			vector<string> best_K_match_imgs(top_K);
			transform(best_K_match_imgs_idx.begin(), best_K_match_imgs_idx.end(), best_K_match_imgs.begin(),
				[&target_img_lst](int i) { return target_img_lst[i]; });


			Mat best_img;
			string best_img_path;
			Mat best_H;

			tie(best_img, best_img_path, best_H) = spatial_verification(input_img, best_K_match_imgs, method);

			return make_tuple(best_img, best_img_path, best_H, best_K_match_imgs);
		}


		void save(const string& db_name) {
			ofstream file(db_name, ios::binary);

			// C++ does not have a direct equivalent to Python's pickle module
			// This might not work as expected with complex types
			file.write((char*)&data_path, sizeof(data_path));
			file.write((char*)&num_imgs, sizeof(num_imgs));
			file.write((char*)&word_to_img, sizeof(word_to_img));
			file.write((char*)&BoW, sizeof(BoW));
			file.write((char*)&word_count, sizeof(word_count));
			file.write((char*)&img_to_histogram, sizeof(img_to_histogram));
			file.write((char*)&all_des, sizeof(all_des));
			file.write((char*)&all_images, sizeof(all_images));
			file.write((char*)&num_feature_per_image, sizeof(num_feature_per_image));
			file.write((char*)&feature_start_idx, sizeof(feature_start_idx));
			file.write((char*)&word_idx_count, sizeof(word_idx_count));
			file.write((char*)&vocabulary_tree, sizeof(vocabulary_tree));

			file.close();
		}


		void load(const string& db_name) {
			ifstream file(db_name, ios::binary);

			// C++ does not have a direct equivalent to Python's pickle module
			// This might not work as expected with complex types
			file.read((char*)&data_path, sizeof(data_path));
			file.read((char*)&num_imgs, sizeof(num_imgs));
			file.read((char*)&word_to_img, sizeof(word_to_img));
			file.read((char*)&BoW, sizeof(BoW));
			file.read((char*)&word_count, sizeof(word_count));
			file.read((char*)&img_to_histogram, sizeof(img_to_histogram));
			file.read((char*)&all_des, sizeof(all_des));
			file.read((char*)&all_images, sizeof(all_images));
			file.read((char*)&num_feature_per_image, sizeof(num_feature_per_image));
			file.read((char*)&feature_start_idx, sizeof(feature_start_idx));
			file.read((char*)&word_idx_count, sizeof(word_idx_count));
			file.read((char*)&vocabulary_tree, sizeof(vocabulary_tree));

			file.close();
		}

		void buildDatabase(string load_path, int k, int L, string method, string save_path) {
			cout << "Initial the Database\n";
			// Database is already initialized in constructor

			cout << "Loading the images from " << load_path << ", use " << method << " for features\n";
			loadImgs(load_path, method);

			cout << "Building Vocabulary Tree, with " << k << " clusters, " << L << " levels\n";
			run_KMeans(k, L);

			cout << "Building Histogram for each images\n";
			build_histogram(vocabulary_tree);

			cout << "Building BoW for each images\n";
			build_BoW();

			cout << "Saving the database to " << save_path << "\n";
			save(save_path);
		}

	};

	int main(int argc, char* argv[]) {

		//Define the test path and DVD cover path
		string test_path = "../data/test";
		string cover_path = "../data/DVDcovers";

		// Initial and build the database
		Database db;

		// Build database
		cout << "Building the database...\n";
		db.buildDatabase(cover_path, 5, 5, "SIFT", "data_sift.txt");

		// Load the database
		cout << "Loading the database...\n";
		db.load("data_sift.txt");

		// Query an image
		string img_path = test_path + "/image_01.jpeg";
		Mat test = imread(img_path);
		auto [best_img, best_img_path, best_H, best_K] = db.query(test, 10, "SIFT");

		return 0;
	}






