/**
 * @file utils.cpp
 * @brief Some useful functions for displaying and matching features
 * @date May 24, 2014
 * @author Pablo F. Alcantarilla
 */

#include "utils.h"
#include <fstream>

using namespace std;

/* ************************************************************************* */
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts) {

  int x = 0, y = 0;
  float radius = 0.0;

  for (size_t i = 0; i < kpts.size(); i++) {
    x = (int)(kpts[i].pt.x+.5);
    y = (int)(kpts[i].pt.y+.5);
    radius = kpts[i].size/2.0;
    cv::circle(img, cv::Point(x,y), 2.5*radius, cv::Scalar(0,255,0), 1);
    cv::circle(img, cv::Point(x,y), 1.0, cv::Scalar(0,0,255), -1);
  }
}

/* ************************************************************************* */
cv::Mat read_homography(const std::string& homography_path) {

  float h11 = 0.0, h12 = 0.0, h13 = 0.0;
  float h21 = 0.0, h22 = 0.0, h23 = 0.0;
  float h31 = 0.0, h32 = 0.0, h33 = 0.0;
  int  tmp_buf_size = 256;
  char tmp_buf[tmp_buf_size];

  // Allocate memory for the OpenCV matrices
  cv::Mat H1toN = cv::Mat::zeros(3, 3, CV_32FC1);

  ifstream infile;
  infile.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit );
  infile.open(homography_path.c_str(),ifstream::in);

  infile.getline(tmp_buf,tmp_buf_size);
  sscanf(tmp_buf,"%f %f %f", &h11, &h12, &h13);

  infile.getline(tmp_buf, tmp_buf_size);
  sscanf(tmp_buf,"%f %f %f", &h21, &h22, &h23);

  infile.getline(tmp_buf, tmp_buf_size);
  sscanf(tmp_buf,"%f %f %f", &h31, &h32, &h33);

  infile.close();

  H1toN.at<float>(0,0) = h11 / h33;
  H1toN.at<float>(0,1) = h12 / h33;
  H1toN.at<float>(0,2) = h13 / h33;

  H1toN.at<float>(1,0) = h21 / h33;
  H1toN.at<float>(1,1) = h22 / h33;
  H1toN.at<float>(1,2) = h23 / h33;

  H1toN.at<float>(2,0) = h31 / h33;
  H1toN.at<float>(2,1) = h32 / h33;
  H1toN.at<float>(2,2) = h33 / h33;
  return H1toN;
}

/* ************************************************************************* */
void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         std::vector<cv::Point2f>& pmatches, const float& nndr) {

  float dist1 = 0.0, dist2 = 0.0;
  for (size_t i = 0; i < matches.size(); i++) {
    cv::DMatch dmatch = matches[i][0];
    dist1 = matches[i][0].distance;
    dist2 = matches[i][1].distance;

    if (dist1 < nndr*dist2) {
      pmatches.push_back(train[dmatch.queryIdx].pt);
      pmatches.push_back(query[dmatch.trainIdx].pt);
    }
  }
}

/* ************************************************************************* */
void compute_inliers_homography(const std::vector<cv::Point2f>& matches,
                                std::vector<cv::Point2f>& inliers,
                                const cv::Mat& H, const float h_max_error) {

  float h11 = 0.0, h12 = 0.0, h13 = 0.0;
  float h21 = 0.0, h22 = 0.0, h23 = 0.0;
  float h31 = 0.0, h32 = 0.0, h33 = 0.0;
  float x1 = 0.0, y1 = 0.0;
  float x2 = 0.0, y2 = 0.0;
  float x2m = 0.0, y2m = 0.0;
  float dist = 0.0, s = 0.0;

  h11 = H.at<float>(0,0);
  h12 = H.at<float>(0,1);
  h13 = H.at<float>(0,2);
  h21 = H.at<float>(1,0);
  h22 = H.at<float>(1,1);
  h23 = H.at<float>(1,2);
  h31 = H.at<float>(2,0);
  h32 = H.at<float>(2,1);
  h33 = H.at<float>(2,2);

  inliers.clear();

  for (size_t i = 0; i < matches.size(); i+=2) {
    x1 = matches[i].x;
    y1 = matches[i].y;
    x2 = matches[i+1].x;
    y2 = matches[i+1].y;

    s = h31*x1 + h32*y1 + h33;
    x2m = (h11*x1 + h12*y1 + h13) / s;
    y2m = (h21*x1 + h22*y1 + h23) / s;
    dist = sqrt( pow(x2m-x2,2) + pow(y2m-y2,2));

    if (dist <= h_max_error) {
      inliers.push_back(matches[i]);
      inliers.push_back(matches[i+1]);
    }
  }
}


/* ************************************************************************* */
void draw_inliers(const cv::Mat& img1, const cv::Mat& imgN, cv::Mat& img_com,
                  const std::vector<cv::Point2f>& ptpairs) {

  int x1 = 0, y1 = 0, xN = 0, yN = 0;
  float rows1 = 0.0, cols1 = 0.0;
  float rowsN = 0.0, colsN = 0.0;
  float ufactor = 0.0, vfactor = 0.0;

  rows1 = img1.rows;
  cols1 = img1.cols;
  rowsN = imgN.rows;
  colsN = imgN.cols;
  ufactor = (float)(cols1)/(float)(colsN);
  vfactor = (float)(rows1)/(float)(rowsN);

  // This is in case the input images don't have the same resolution
  cv::Mat img_aux = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
  cv::resize(imgN, img_aux, cv::Size(img1.cols, img1.rows), 0, 0, cv::INTER_LINEAR);

  for (int i = 0; i < img_com.rows; i++) {
    for (int j = 0; j < img_com.cols; j++) {
      if (j < img1.cols) {
        *(img_com.ptr<unsigned char>(i)+3*j) = *(img1.ptr<unsigned char>(i)+3*j);
        *(img_com.ptr<unsigned char>(i)+3*j+1) = *(img1.ptr<unsigned char>(i)+3*j+1);
        *(img_com.ptr<unsigned char>(i)+3*j+2) = *(img1.ptr<unsigned char>(i)+3*j+2);
      }
      else {
        *(img_com.ptr<unsigned char>(i)+3*j) = *(imgN.ptr<unsigned char>(i)+3*(j-img_aux.cols));
        *(img_com.ptr<unsigned char>(i)+3*j+1) = *(imgN.ptr<unsigned char>(i)+3*(j-img_aux.cols)+1);
        *(img_com.ptr<unsigned char>(i)+3*j+2) = *(imgN.ptr<unsigned char>(i)+3*(j-img_aux.cols)+2);
      }
    }
  }

  for (size_t i = 0; i < ptpairs.size(); i+= 2) {
    x1 = (int)(ptpairs[i].x+.5);
    y1 = (int)(ptpairs[i].y+.5);
    xN = (int)(ptpairs[i+1].x*ufactor+img1.cols+.5);
    yN = (int)(ptpairs[i+1].y*vfactor+.5);
    cv::line(img_com, cv::Point(x1,y1), cv::Point(xN,yN), cv::Scalar(255,0,0), 2);
  }
}
