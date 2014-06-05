/**
 * @file test_akaze_match.cpp
 * @brief Main program for testing OpenCV A-KAZE port in an image matching application
 * @date Jun 05, 2014
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

  if (argc != 4) {
    cerr << "Error introducing input arguments!" << endl;
    cerr << "The format needs to be: ./test_akaze_match img1 imgN H1toN" << endl;
    return -1;
  }

  cv::Mat img1, imgN;
  string img1File = argv[1];
  string imgNFile = argv[2];
  string HFile = argv[3];

  // Open the input image
  img1 = imread(img1File, 1);
  imgN = imread(imgNFile, 1);
  cv::Mat H1toN = read_homography(HFile);

  // Create A-KAZE object
  cv::KAZE dkaze;

  // Timing information
  double t1 = 0.0, t2 = 0.0;
  double tkaze = 0.0, tmatch = 0.0;

  // Detect KAZE features in the images
  vector<cv::KeyPoint> kpts1, kptsN;
  cv::Mat desc1, descN;

  t1 = cv::getTickCount();
  dkaze(img1, cv::noArray(), kpts1, desc1);
  dkaze(imgN, cv::noArray(), kptsN, descN);
  t2 = cv::getTickCount();
  tkaze = 1000.0*(t2-t1) / cv::getTickFrequency();

  int nr_kpts1 = kpts1.size();
  int nr_kptsN = kptsN.size();

  // Match the descriptors using NNDR matching strategy
  vector<vector<cv::DMatch> > dmatches;
  vector<cv::Point2f> matches, inliers;
  cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce");
  float nndr = 0.8;

  t1 = cv::getTickCount();
  matcher_l2->knnMatch(desc1, descN, dmatches, 2);
  matches2points_nndr(kpts1, kptsN, dmatches, matches, nndr);
  t2 = cv::getTickCount();
  tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Compute the inliers using the ground truth homography
  float max_h_error = 2.5;
  compute_inliers_homography(matches, inliers, H1toN, max_h_error);

  // Compute the inliers statistics
  int nr_matches = matches.size()/2;
  int nr_inliers = inliers.size()/2;
  int nr_outliers = nr_matches - nr_inliers;
  float ratio = 100.0*((float) nr_inliers / (float) nr_matches);

  cout << "KAZE Matching Results" << endl;
  cout << "*******************************" << endl;
  cout << "# Keypoints 1:                        \t" << nr_kpts1 << endl;
  cout << "# Keypoints N:                        \t" << nr_kptsN << endl;
  cout << "# Matches:                            \t" << nr_matches << endl;
  cout << "# Inliers:                            \t" << nr_inliers << endl;
  cout << "# Outliers:                           \t" << nr_outliers << endl;
  cout << "Inliers Ratio (%):                    \t" << ratio << endl;
  cout << "Time Detection+Description (ms):      \t" << tkaze << endl;
  cout << "Time Matching (ms):                   \t" << tmatch << endl;
  cout << endl;
  return 1;
}
