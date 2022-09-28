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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat &matPos,Map* map): 
    mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), 
    mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0), 
    mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(static_cast<MapPoint*>(NULL)), 
    mfMinDistance(0), mfMaxDistance(0), mpMap(map)
{
     Pos.copyTo(mWorldPos);
     mNormalVector = cv::Mat::zeros(3,1,CV_32F);
    
     unique_lock<mutex> lock(mpMap->mMutexPointCreation);
     mnId = nNextId++;
}
    
KeyFrame* MapPoint::SetKeyFrame(KeyFrame* keyFrame)
{
    return mpRefKF = keyFrame;
}

    
/**
 * @brief Construct MapPoint given coordinates and keyframe
 *
 * binocular：StereoInitialization()，CreateNewKeyFrame()，LocalMapping::CreateNewMapPoints()
 * Monocular：CreateInitialMapMonocular()，LocalMapping::CreateNewMapPoints()
 * @param Pos    The coordinates of the MapPoint (wrt world coordinate system)
 * @param pRefKF KeyFrame
 * @param pMap   Map
 */
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

/**
 * @brief Construct MapPoint given coordinates and frame
 *
 * binocular：UpdateLastFrame()
 * @param Pos    The coordinates of the MapPoint (wrt world coordinate system)
 * @param pMap   Map
 * @param pFrame Frame
 * @param idxF   The index of MapPoint in the Frame, that is, the number of the corresponding feature point
 */
MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;// The vector from the camera to the 3D point in the world coordinate system
    mNormalVector = mNormalVector/cv::norm(mNormalVector);// The unit vector from the camera to the 3D point in the world coordinate system

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    // See also the note before the PredictScale function
    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    // See mDescriptor's comment in MapPoint.h
    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

/**
 * @brief Add observations
 *
 * Record which feature points of KeyFrame can observe the MapPoint \n
 * And increase the number of observed cameras nObs, monocular+1, binocular or grbd+2
 * This function is the core function for establishing the co-view relationship of key frames. The key frames that can jointly observe some MapPoints are co-view key frames.
 * @param pKF KeyFrame
 * @param idx Index of MapPoint in KeyFrame
 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    // Record the KF that can observe the MapPoint and the index of the MapPoint in the KF
    mObservations[pKF]=idx;

    if(pKF->mvuRight[idx]>=0)
        nObs+=2;// binocular or grbd
    else
        nObs++;// monocular
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            // If the keyFrame is a reference frame, re-specify RefFrame after the Frame is deleted
            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            // When the number of cameras that observe the point is less than 2, discard the point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

// Tell the Frame that can observe the MapPoint, the MapPoint has been deleted
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;// Dump mObservations to obs, pointers are stored in obs and mObservations, and the assignment process is shallow copy
        mObservations.clear();// Release the memory pointed to by mObservations, and automatically delete obs as a local variable
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);// Tell the KeyFrame that can observe the MapPoint, the MapPoint was deleted
    }

    mpMap->EraseMapPoint(this);// Erase the memory requested by the MapPoint
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

// When forming a closed loop, the relationship between KeyFrame and MapPoint will be updated
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;// This section is the same as the SetBadFlag function
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    // All keyframes that can observe the MapPoint must be replaced
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);// Let KeyFrame replace the original MapPoint with pMP
            pMP->AddObservation(pKF,mit->second);// Let MapPoint replace the corresponding KeyFrame
        }
        else
        {
            // A conflict occurs, that is, there are two feature points a, b in pKF (the descriptors of these two feature points are approximately the same), and these two feature points correspond to two MapPoints as this, pMP
            // However, in the process of fuse, there are more observations of pMP, and this needs to be replaced, so the connection between b and pMP is retained, and the connection between a and this is removed
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

// MapPoints not detected by MapPointCulling
bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

/**
 * @brief Increase Visible
 *
 * Visible said:
 * 1. The MapPoint is within the field of view of certain frames, judged by the Frame::isInFrustum() function
 * 2. The MapPoint is observed in these frames, but it does not necessarily match the feature points of these frames
 * For example: there is a MapPoint (denoted as M), within the field of view of a certain frame F,
 * But it does not mean that the point M can be matched with a feature point in the frame of F
 */
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

/**
 * @brief Increase Found
 *
 * The number of frames in which the point can be found + n, n defaults to 1
 * @see Tracking::TrackLocalMap()
 */
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

/**
 * @brief Compute a descriptor with a representative
 *
 * Since a MapPoint will be observed by many cameras, after inserting keyframes, it is necessary to determine whether to update the most suitable descriptor for the current point \n
 * First obtain all descriptors of the current point, and then calculate the pairwise distance between the descriptors, the best descriptor and other descriptors should have the smallest median distance
 * @see III - C3.3
 */
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    // Traverse all keyframes where the 3d point is observed, get the orb descriptor, and insert it into vDescriptors
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    // Get the distance between these descriptors
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        // distance from the ith descriptor to all other descriptors
        //vector<int> vDists(Distances[i],Distances[i]+N);
        vector<int> vDists(Distances[i].begin(), Distances[i].end());
        sort(vDists.begin(),vDists.end());
        
        // get the median
        int median = vDists[0.5*(N-1)];

        // find the smallest median
        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        // the best descriptor, which has the smallest median distance to other descriptors
        // Simplified, the median represents the average distance from this descriptor to other descriptors
        // The best descriptor is the one with the smallest average distance from other descriptors
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

/**
 * @brief check MapPoint is in keyframe
 * @param  pKF KeyFrame
 * @return     true if in pKF
 */
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

/**
 * @brief Update the average observation direction and observation distance range
 *
 * Since a MapPoint will be observed by many cameras, after inserting keyframes, the corresponding variables need to be updated
 * mNormalVector：The average direction in which the 3D point is observed
 * mfMaxDistance：The maximum distance to observe the 3D point
 * mfMinDistance：Minimum distance to observe this 3D point
 * @see III - C2.2 c2.4
 */
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;// Get all keyframes where the 3d point is observed
        pRefKF=mpRefKF;// Observe the reference keyframe for this point
        Pos = mWorldPos.clone();// The position of the 3d point in the world coordinate system
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);// Normalize the observation direction of the point to a unit vector for all keyframes and sum up
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();// Reference keyframe camera points to the vector of the 3D point (represented in the world coordinate system)
    const float dist = cv::norm(PC);// The distance from this point to the reference keyframe camera
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;// pyramid levels

    {
        unique_lock<mutex> lock3(mMutexPos);
        // See also the note before the PredictScale function
        mfMaxDistance = dist*levelScaleFactor;// Observe the maximum distance from the point
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];// Observe the minimum distance from the point
        mNormalVector = normal/n;// get the average viewing direction
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

//              ____
// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)
//            log(1.2)
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor is the distance after considering the upper scale for the reference frame
        ratio = mfMaxDistance/currentDist;
    }

    // Simultaneously take log linearization
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

// map serialization addition
MapPoint::MapPoint():
    nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0),mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0)
{}
template<class Archive>
void MapPoint::serialize(Archive &ar, const unsigned int version)
{
    unique_lock<mutex> lock_Pos(mMutexPos);
    unique_lock<mutex> lock_Features(mMutexFeatures);
    ar & mnId & nNextId & mnFirstKFid & mnFirstFrame & nObs;
    // Tracking related vars
    ar & mTrackProjX;
    ar & mTrackProjY;
    ar & mTrackProjXR;
    ar & mbTrackInView;
    ar & mnTrackScaleLevel;
    ar & mTrackViewCos;
    ar & mnTrackReferenceForFrame;
    ar & mnLastFrameSeen;
    // Local Mapping related vars
    ar & mnBALocalForKF & mnFuseCandidateForKF;
    // Loop Closing related vars
    ar & mnLoopPointForKF & mnCorrectedByKF & mnCorrectedReference & mPosGBA & mnBAGlobalForKF;
    // don't save the mutex
    ar & mWorldPos;
    ar & mObservations;
    ar & mNormalVector;
    ar & mDescriptor;
    ar & mpRefKF;
    ar & mnVisible & mnFound;
    ar & mbBad & mpReplaced;
    ar & mfMinDistance & mfMaxDistance;
    ar & mpMap;
    // don't save the mutex
}
template void MapPoint::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void MapPoint::serialize(boost::archive::binary_oarchive&, const unsigned int);

} //namespace ORB_SLAM
