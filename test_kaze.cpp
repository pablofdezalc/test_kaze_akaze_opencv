/**
 * @file test_kaze_features_port.cpp
 * @brief Main program for testing OpenCV KAZE port
 * @date May 24, 2014
 * @author Pablo F. Alcantarilla
 */

#include "./src/utils.h"

// System
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

/* ************************************************************************* */
int main(int argc, char *argv[]) {

	cv::Mat img;

	if (argc != 2) {
		cerr << "Error introducing input arguments!" << endl;
		cerr << "The format needs to be: ./test_kaze_features_port img" << endl;
		return -1;
	}

	string imgFile = argv[1];

	// Open the input image
	img = imread(imgFile, 1);

	// Create KAZE object
	cv::KAZE dkaze;

	// Timing information
	double t1 = 0.0, t2 = 0.0;
	double tkaze = 0.0;

	// Detect KAZE features in the input image
	vector<cv::KeyPoint> keypoints;

	t1 = cv::getTickCount();
	dkaze(img, cv::noArray(), keypoints);
	t2 = cv::getTickCount();
	tkaze = 1000.0*(t2-t1) / cv::getTickFrequency();

	draw_keypoints(img, keypoints);

	// Show the detected KAZE features
	cv::imshow("KAZE", img);
	cv::waitKey(0);

	int nr_keypoints = keypoints.size();

	cout << "KAZE Results" << endl;
	cout << "********************" << endl;
	cout << "# Keypoints:    \t" << nr_keypoints << endl;
	cout << "Time (ms):      \t" << tkaze << endl;
	cout << endl;

	return 0;
}
