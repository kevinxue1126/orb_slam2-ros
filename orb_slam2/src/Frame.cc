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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

/**
 * @brief Copy constructor
 *
 * copy constructor, mLastFrame = Frame(mCurrentFrame)
 */
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


// Dual purpose initialization
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // Simultaneous feature extraction of left and right eyes
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    // Undistort feature points, no binocular correction is performed here, because the input image is required to be polar corrected
    UndistortKeyPoints();

    // Calculate the matching between binoculars, and the depth of the successfully matched feature points will be calculated
    // Depth is stored in mvuRight and mvDepth
    ComputeStereoMatches();

    // Corresponding mappoints
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// RGBD initialization
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// Monocular initialization
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    // Call the correction function of OpenCV to correct the feature points extracted by orb
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // Each feature point is recorded in mGrid
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

/**
 * @brief Set the camera pose.
 * 
 * Set the camera pose, then call UpdatePoseMatrices() to change the values of mRcw, mRwc and other variables
 * @param Tcw Transformation from world to camera
 */
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

/**
 * @brief Computes rotation, translation and camera center matrices from the camera pose.
 *
 * Calculate mRcw, mtcw and mRwc, mOw from Tcw
 */
void Frame::UpdatePoseMatrices()
{ 
    // [x_camera 1] = [R|t]*[x_world 1]，Coordinates are in homogeneous form
    // x_camera = R*x_world + t
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    // mtcw, that is, the vector between the camera coordinate system and the world coordinate system in the camera coordinate system, and the vector direction points from the camera coordinate system to the world coordinate system
    // mOw, that is, the vector between the world coordinate system and the camera coordinate system in the world coordinate system, and the vector direction points from the world coordinate system to the camera coordinate system
    mOw = -mRcw.t()*mtcw;
}

/**
 * @brief Determine if a point is in view
 *
 * Calculated reprojection coordinates, angle of observation direction, and predicted scale in the current frame
 * @param  pMP             MapPoint
 * @param  viewingCosLimit Orientation thresholds for viewing angle and average viewing angle
 * @return                 true if is in view
 * @see SearchLocalPoints()
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    // The coordinates of the 3D point P in the camera coordinate system
    const cv::Mat Pc = mRcw*P+mtcw;// The Rt here is after preliminary optimization
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    // V-D 1) Project the MapPoint to the current frame and determine whether it is in the image
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // V-D 3) Calculate the distance from MapPoint to the center of the camera, and determine whether it is within the distance of the scale change
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    // In the world coordinate system, the vector from the camera to the 3D point P, the vector direction is from the camera to the 3D point P
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    // V-D 2) Calculate the cosine value of the angle between the current viewing angle and the average viewing angle, if it is less than cos(60), that is, the angle is greater than 60 degrees, return
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // V-D 4) Predict the scale according to the depth (corresponding feature points are in one layer)
    // ! ! ! ! It can be found through the function call that isInFrustum will only be called by the current mCurrentFrame in Tracking.cpp
    // Therefore, nPredictedLevel records the possible observed pyramid levels of the 3D point on mCurrentFrame
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    // mark the point to be projected in the future
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;//The abscissa of the 3D point projected to the right camera of the binocular
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

/**
 * @brief Find the feature points in the square with x, y as the center, side length 2r and in [minLevel, maxLevel]
 * @param x        image coordinates u
 * @param y        image coordinates v
 * @param r        side length
 * @param minLevel minimum scale
 * @param maxLevel maximum scale
 * @return         The sequence number of the feature point that satisfies the condition
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


/**
 * @brief Bag of Words Representation
 *
 * Calculate the word bags mBowVec and mFeatVec, where mFeatVec records ni descriptors belonging to the ith node (at layer 4)
 * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
 */
void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

// Call the correction function of OpenCV to correct the feature points extracted by orb
void Frame::UndistortKeyPoints()
{
    // If no image is rectified, no distortion
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    //N is the number of extracted feature points, save N feature points in N*2 mat
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // Adjust the channel of mat to 2, and the shape of the rows and columns of the matrix remains unchanged
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // Store the corrected feature points
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // Correct the first four boundary points: (0,0) (cols,0) (0,rows) (cols,rows)
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; //upper left
        mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; //top right
        mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; //lower left
        mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; //lower right
        mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));//The smallest abscissa of the upper left and lower left
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));//The upper-right and lower-right abscissas have the largest
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));//The smallest vertical coordinates of the upper left and upper right
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));//The lower left and lower right vertical coordinates are the smallest

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * @brief binocular matching
 *
 * Find matching points in the right image for each feature point in the left image \n
 * Find a match according to the descriptor distance on the baseline (with redundant range), and then perform SAD precise positioning \n
 * Finally, sort all SAD values, eliminate matching pairs with larger SAD values, and then use parabolic fitting to obtain sub-pixel precision matching \n
 * After the match is successful, mvuRight(ur) and mvDepth(Z) will be updated
 */
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    // Step 1: Establish a feature point search range correspondence table, a feature point searches for matching feature points in a strip area
    // When searching for matching, it is not only searched on a horizontal line, but on a horizontal search strip. In short, the original ordinate of each feature point is 1. Here, the volume of the feature point is enlarged, and the ordinate is take up several lines
    // For example, the ordinate of a feature point in the left eye image is 20, then when searching on the right image, the search is performed on the band whose ordinate is 18 to 22. The width of the search band is plus or minus 2. The width of the search band and the feature point related to the level of the pyramid
    // To put it simply, if the ordinate is 20 and the feature point is in the 20th line of the image, then it is considered that lines 18 19 20 21 22 all have this feature point
    // vRowIndices[18], vRowIndices[19], vRowIndices[20], vRowIndices[21], vRowIndices[22] all have this feature point number
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        // !! In this function, the binocular correction is not performed, and the binocular correction is implemented in the outer program
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // Calculate the vertical width of the matching search, the larger the scale (the higher the number of layers, the closer the distance), the larger the search range
        // If the feature point is in the first layer of the pyramid, the search range is: plus or minus 2
        // The larger the scale, the higher the location uncertainty, so the larger the search radius
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb; // NOTE bug mb is not initialized, and the assignment of mb is placed after the ComputeStereoMatches function in the constructor
    const float minD = 0; // Minimum parallax, set to 0
    const float maxD = mbf/minZ;// Maximum parallax, corresponding to minimum depth mbf/minZ = mbf/mb = mbf/(mbf/fx) = fx

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // Step 2: For each feature point of the left eye camera, find the matching point in the right eye band search area through the descriptor, and then do sub-pixel matching through SAD
    // Note: here are mvKeys before correction, not mvKeysUn after correction
    // Inconsistency between KeyFrame::UnprojectStereo and Frame::UnprojectStereo functions
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // possible matching points
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;// Minimum match range
        const float maxU = uL-minD;// Maximum match range

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        // Each feature point descriptor occupies one line, and a pointer is established to point to the descriptor corresponding to the iL feature point
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // Step 2.1: Traverse all possible matching points in the right eye to find the best matching point (minimum descriptor distance)
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // Match only feature points at the nearest neighbor scale
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }
        // The matching error of the best match exists in bestDist, and the position of the matching point exists in bestIdxR

        // Subpixel match by correlation
        // Step 2.2: Improve pixel matching correction bestincR by SAD matching
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            //kpL.pt.x corresponds to the coordinates of the bottom layer of the pyramid, and scales the best matching feature point pair to the corresponding scale layer (scaleduL, scaledvL) (scaleduR0, )
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;// The size of the sliding window is 11*11 Note that the window is taken from the resized image
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);//Each element in the window is subtracted from the element in the center, which is simply normalized to reduce the influence of light intensity
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            // The sliding range of the sliding window is (-L, L), and it is judged in advance whether the sliding window will cross the boundary during the sliding process.
            const float iniu = scaleduR0+L-w;//Should this place be scaleduR0-L-w
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                // Horizontal sliding window
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);//Each element in the window is subtracted from the element in the center, which is simply normalized to reduce the influence of light intensity

                float dist = cv::norm(IL,IR,cv::NORM_L1);// One norm, computes the absolute value of the difference
                if(dist<bestDist)
                {
                    bestDist =  dist;// SAD matching current minimum matching deviation
                    bestincR = incR;// SAD matches the current best modifier
                }

                vDists[L+incR] = dist;// Under normal circumstances, the data in this should change in a parabolic form
            }

            if(bestincR==-L || bestincR==L)// During the entire sliding window process, the minimum value of SAD does not appear in the form of a parabola, the SAD matching fails, and the depth of the feature point is abandoned.
                continue;

            // Sub-pixel match (Parabola fitting)
            // Step 2.3: Do a parabolic fit to find the valley bottom to get the sub-pixel matching deltaR
            // (bestincR,dist) (bestincR-1,dist) (bestincR+1,dist) three points fit a parabola
            // bestincR+deltaR is the position of the parabolic valley bottom, and the correction amount of bestincR relative to the minimum value matched by SAD is deltaR
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // The correction amount obtained by parabolic fitting cannot exceed one pixel, otherwise the depth of the feature point will be discarded
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // Through descriptor matching, the matching point position is scaleduR0
            // Find the correction bestincR by SAD matching
            // Find the subpixel correction deltaR by parabolic fitting
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            // here is the disparity, according to which the depth is calculated
            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)// Finally determine whether the parallax is within the range
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // depth is calculated here
                // depth=baseline*fx/disparity
                mvDepth[iL]=mbf/disparity;// depth
                mvuRight[iL] = bestuR;// The matching pair is on the abscissa of the right image
                vDistIdx.push_back(pair<int,int>(bestDist,iL));// The feature point SAD matches the minimum matching deviation
            }
        }
    }

    // Step 3: Eliminate matching feature points with large SAD matching deviation
    // The previous SAD matching only determines whether there is a local minimum in the sliding window. Here, the depth of the feature points with large SAD matching deviation is eliminated by comparison.
    sort(vDistIdx.begin(),vDistIdx.end());// Sort according to the SAD deviation of all matching pairs, the distance is from small to large
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;// Calculate the adaptive distance, matching pairs larger than this distance will be eliminated

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    // mvDepth is read directly from the depth image
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i i-th keypoint
 * @return   3D point (relative to the world coordinate system)
 */
cv::Mat Frame::UnprojectStereo(const int &i)
{
    // KeyFrame::UnprojectStereo
    // mvDepth is obtained in the ComputeStereoMatches function
    // The feature point before correction corresponding to mvDepth, but here is the back projection of the corrected feature point
    // KeyFrame::UnprojectStereo is the back projection of the feature point mvKeys before correction
    // In the ComputeStereoMatches function, the depth of the corrected feature points should be calculated
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
