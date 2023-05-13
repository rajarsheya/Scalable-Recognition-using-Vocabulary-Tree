#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

const int SCALE = 2;

/** Function to display the input image in a resized window
* Preconditions: 1. The input Mat is non-empty
*                2. The resizing SCALE is already initialized
* Postconditions: 1. The input image is displayed in a resized window named windowName
*				  2. A key press is performed to complete the function execution
* Assumptions: The resized window can fit the larger image within the screen limits
*/
void displayResult(String windowName, Mat image)
{
	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, image.cols / SCALE, image.rows / SCALE);    
	imshow(windowName, image);
	waitKey(0);
}

//Finds the most occuring cluster for each descriptor
int maxVotes(vector<int>v)
{
	map<int, int> freq;
	for (int x : v)
	{
		freq[x]++;
	}
	int max_value = 0;
	vector<int> max_keys;
	for (auto p : freq)
	{
		if (p.second > max_value)
		{
			max_value = p.second;
			max_keys.clear();
			max_keys.push_back(p.first);
		}
		else if (p.second == max_value)
		{
			max_keys.push_back(p.first);
		}
	}
	int index = rand() % max_keys.size();
	return max_keys[index];
}

void formKTree(vector<Mat> images)
{
	vector<Mat> descriptors;
	for (int i=0;i<images.size();i++)
	{
		/*
		string name = "image" + to_string(i + 1);
		displayResult(name, images[i]);
		*/
		//r<Feature2D>  orb = ORB::create();
		vector<KeyPoint> keypoint;
		Mat descriptor;
		//b->detectAndCompute(images[i], noArray(), keypoint, descriptor);
		//scriptors.push_back(descriptor);
		Ptr<FeatureDetector> orbDetector = SIFT::create();
		orbDetector->detect(images[i], keypoint);
		Ptr<DescriptorExtractor> orbExtractor = SIFT::create();
		orbExtractor->compute(images[i], keypoint, descriptor);
		descriptors.push_back(descriptor);
	}

	// Converts descriptors to a floating-point matrix and reshape it into a one-dimensional vector
	int rows = 0;
	int cols = descriptors[0].cols;
	for (int i = 0; i < descriptors.size(); i++)
	{
		rows += descriptors[i].rows;
	}
	Mat data(rows, cols, CV_32F);
	int start = 0;
	for (int i = 0; i < descriptors.size(); i++)
	{
		Mat submat = data.rowRange(start, start + descriptors[i].rows);
		descriptors[i].convertTo(submat, CV_32F);
		start += descriptors[i].rows;
	}
	
	// Choose the number of clusters and initialize random centers
	int k = 3;
	Mat labels, centers;
	TermCriteria criteria(TermCriteria::COUNT, 10, 0.0);
	kmeans(data, k, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

	// Assign each descriptor to the nearest center
	vector<vector<int>> clusters(k);
	start = 0;
	for (int i = 0; i < descriptors.size(); i++)
	{
		vector<int>clusterIds;
		int clusterId;
		for (int j = 0; j < descriptors[i].rows; j++)
		{
			clusterIds.push_back(labels.at<int>(start + j));
		}
		clusters[maxVotes(clusterIds)].push_back(i);
		start += descriptors[i].rows;
	}

	// Display or save the images that belong to each cluster
	for (int i = 0; i < k; i++)
	{
		cout << "Cluster " << i << " contains " << clusters[i].size() << " images:" << endl;
		for (int j = 0; j < clusters[i].size(); j++)
		{
			int index = clusters[i][j];
			cout << "Image " << index << endl;
			displayResult("plane" + to_string(index + 1), images[index]);
		}
		cout << "End of cluster" << endl;
	}
}

int main(int argc, char* argv[])
{
	vector<Mat> images;
	for (int i = 1; i <= 10; i++)
	{
		string name = "plane" + to_string(i) + ".jpg";
		Mat image = imread(name);
		images.push_back(image);
	}
	formKTree(images);
	return 0;
}