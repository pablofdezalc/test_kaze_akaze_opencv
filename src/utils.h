/**
 * @file utils.h
 * @brief Some useful functions for displaying and matching features
 * @date May 21, 2014
 * @author Pablo F. Alcantarilla
 */

#pragma once

/* ************************************************************************* */
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

/* ************************************************************************* */
/**
 * @brief This function draws the list of detected keypoints
 * @param img Input image
 * @param kpts Vector of detected keypoints
 */
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);

/**
 * @brief Function for reading the ground truth homography from a txt file
 * @param homography_file Path for the file that contains the ground truth homography
 */
cv::Mat read_homography(const std::string& homography_path);

/**
 * @brief This function converts matches to points using nearest neighbor distance
 * ratio matching strategy
 * @param train Vector of keypoints from the first image
 * @param query Vector of keypoints from the second image
 * @param matches Vector of nearest neighbors for each keypoint
 * @param pmatches Vector of putative matches
 * @param nndr Nearest neighbor distance ratio value
 */
void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         std::vector<cv::Point2f>& pmatches, const float& nndr);

/**
 * @brief This function computes the set of inliers given a ground truth homography
 * @param matches Vector of putative matches
 * @param inliers Vector of inliers
 * @param H Ground truth homography matrix 3x3
 * @param h_max_error The maximum pixel location error to accept an inlier
 */
void compute_inliers_homography(const std::vector<cv::Point2f>& matches,
                                std::vector<cv::Point2f>& inliers,
                                const cv::Mat& H, const float h_max_error);

/**
 * @brief This function draws the set of the inliers between the two images
 * @param img1 First image
 * @param img2 Second image
 * @param img_com Image with the inliers
 * @param ptpairs Vector of point pairs with the set of inliers
 */
void draw_inliers(const cv::Mat& img1, const cv::Mat& imgN, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs);
