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
