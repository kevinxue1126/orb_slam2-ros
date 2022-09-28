/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
#include "Frame.h"


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
/**
 * @brief Monocular SLAM initialization is related, binocular and RGBD will not use this class
 */
class Initializer
{
    typedef pair<int,int> Match;

public:

    // Fix the reference frame
    // Use the reference frame to initialize, this reference frame is the first frame when SLAM officially starts
    Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    // Use the current frame, that is, use the second frame of SLAM logic to initialize the entire SLAM, and get the R t between the first two frames, and the point cloud
    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);


private:
    // Assuming that the scene is flat, obtain the Homography matrix (current frame 2 to reference frame 1) through the first two frames, and get the score of the model
    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    // Assuming that the scene is non-planar, the Fundamental matrix (current frame 2 to reference frame 1) is obtained through the first two frames, and the score of the model is obtained
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);

    // Called by the FindHomography function to calculate the Homography matrix
    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    // Called by the FindFundamental function to calculate the Fundamental matrix
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

    // Called by the FindHomography function to calculate the score assuming the Homography model is used
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
    // Called by the FindFundamental function to calculate the score assuming the Fundamental model is used
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);
    
    // Decompose the F matrix and find the appropriate R, t from the decomposed solutions
    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // Decompose the H matrix and find the appropriate R, t from the decomposed solutions
    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // Through the triangulation method, the feature points are restored to 3D points using the back-projection matrix
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    // Normalized 3D space point and inter-frame displacement t
    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    // ReconstructF calls this function to perform cheirality check, so as to further find the most suitable solution after F decomposition
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    // The F matrix can get the Essential matrix by combining the internal parameters. This function is used to decompose the E matrix, and 4 sets of solutions will be obtained.
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1)
    vector<cv::KeyPoint> mvKeys1;///< Store feature points in Reference Frame

    // Keypoints from Current Frame (Frame 2)
    vector<cv::KeyPoint> mvKeys2;///< Store feature points in Current Frame

    // Current Matches from Reference to Current
    // Reference Frame: 1, Current Frame: 2
    vector<Match> mvMatches12;///< The data structure of Match is pair, and mvMatches12 only records the feature point pair from Reference to Current matching.
    vector<bool> mvbMatched1;///< Record whether each feature point of the Reference Frame has a matching feature point in the Current Frame

    // Calibration
    cv::Mat mK;///< Camera internal parameters

    // Standard Deviation and Variance
    float mSigma, mSigma2;///< Measurement error

    // Ransac max iterations
    int mMaxIterations;///< Number of RANSAC iterations when computing Fundamental and Homography matrices

    // Ransac sets
    vector<vector<size_t> > mvSets;  ///< Two-dimensional container, the size of the outer container is the number of iterations, and the size of the inner container is the point needed to calculate the H or F matrix in each iteration 

};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
