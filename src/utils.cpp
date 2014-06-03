/**
 * @file utils.cpp
 * @brief Some useful functions for displaying and matching features
 * @date May 24, 2014
 * @author Pablo F. Alcantarilla
 */

#include "utils.h"

using namespace std;

/* ************************************************************************* */

void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts) {

	int x = 0, y = 0;
	float radius = 0.0;

	for (size_t i = 0; i < kpts.size(); i++) {
		x = (int)(kpts[i].pt.x+.5);
		y = (int)(kpts[i].pt.y+.5);
		radius = kpts[i].size/2.0;
		cv::circle(img, cv::Point(x,y), 2.5*radius, CV_RGB(0,255,0), 1);
		cv::circle(img, cv::Point(x,y), 1.0, CV_RGB(0,0,255), -1);
	}
}

