//-----------------------------------------------------------------------------------------------------------------------------------------------------------
//
//  Enhanced Vocabulary Trees for Real-Time Object Recognition in Image and Video Streams
//  Team members:  Arsheya Raj, Sugam Jaiswal, Josiah Zacharias
//
//  This project is inspired on the research paper which has the topic as "Scalable Recognition with a Vocabulary Tree". The goal of the project is to
//  find the best match for the query image from a dataset with images in the range of thousands. The project aims to develop a system that can
//  efficiently and accurately retrieve an image from a large collection of images based on a given query image. The system should be able to
//  compare the query image with thousands of images in the dataset and return the one that is most similar to it in terms of visual features,
//  such as color, shape, texture, or content. The project will explore various methods and techniques for image representation, score calculation and 
//  image indexing.
// 
//   Paper Link: https://ieeexplore-ieee-org.offcampus.lib.washington.edu/document/1641018
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

#include "FeatureDetector1.h"
#include "VocabNode.h"
#include "Homography.h"
#include "Database.h"

using namespace std;
using namespace cv;
//mespace fs = __fs::filesystem; // for MacOS
namespace fs = std::experimental::filesystem; // for Windows
using namespace fs;

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
    string cover_path = "./data/DVD_DB_50";

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

    string img_path = test_path + "/query_03.jpg";

    Mat test = imread(img_path);
    vector<string> best_img_path;
    vector<cv::String> best_K;
    Database db;

    // Build database
    cout << "Building the database...\n";
    chrono::time_point<std::chrono::high_resolution_clock> startdbbuild, enddbbuild;
    startdbbuild = std::chrono::high_resolution_clock::now();
    db.buildDatabase(cover_path, 3, 5, fdname);
    enddbbuild = std::chrono::high_resolution_clock::now();
    chrono::duration< double > Time_for_db_build = enddbbuild - startdbbuild;
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
