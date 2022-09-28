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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"
#include "BoostArchiver.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;

/**
 * @brief MapPoint is a map point
 */
class MapPoint
{
public:
    
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);
    
    MapPoint(const cv::Mat &matPos,Map* map);

    KeyFrame* SetKeyFrame(KeyFrame* keyFrame);
    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId; ///< Global ID for MapPoint
    static long unsigned int nNextId;
    long int mnFirstKFid; ///< Create the keyframe ID of this MapPoint
    long int mnFirstFrame;///< Create the frame ID of the MapPoint (that is, each keyframe has a frame ID)
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    // TrackLocalMap - The variable in SearchByProjection that decides whether to project the point or not
    // There are several points of mbTrackInView==false:
    // a has been matched with the current frame (TrackReferenceKeyFrame, TrackWithMotionModel) but is considered to be an outlier during the optimization process
    // b has been matched with the current frame and is an interior point, and such points do not need to be projected again
    // c Points that are not in the current camera field of view (that is, not judged by isInFrustum)
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    // TrackLocalMap - Flag in UpdateLocalPoints to prevent duplicate addition of MapPoints to mvpLocalMapPoints
    long unsigned int mnTrackReferenceForFrame;
    // TrackLocalMap - The variable in SearchLocalPoints that determines whether to perform isInFrustum judgment
    // There are several points where mnLastFrameSeen==mCurrentFrame.mnId:
    // a has been matched with the current frame (TrackReferenceKeyFrame, TrackWithMotionModel) but is considered to be an outlier during the optimization process
    // b has been matched with the current frame and is an interior point, and such points do not need to be projected again
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;///< The coordinates of the MapPoint in the world coordinate system

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;///< Observe the KF of the MapPoint and the index of the MapPoint in KF

     // Mean viewing direction
     // The MapPoint mean viewing direction
     cv::Mat mNormalVector;

     // Best descriptor to fast matching
     // The MapPoint mean viewing direction
     // If MapPoint corresponds to many frame image feature points (when constructed by keyframe), then the descriptor with the smallest average distance from other descriptors is the best descriptor
     // MapPoint only corresponds to the image feature point of one frame (when constructed by frame), then the descriptor of this feature point is the descriptor of the 3D point
     cv::Mat mDescriptor;///< The optimal descriptor obtained by ComputeDistinctiveDescriptors()

     // Reference KeyFrame
     KeyFrame* mpRefKF;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;

// map serialization addition
public:
    // for serialization
    MapPoint();
private:
    // serialize is recommended to be private
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version);
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
