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
