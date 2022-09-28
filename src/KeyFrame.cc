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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(KeyFrameInitialization &kfInit, Map *map, KeyFrameDatabase *kfDB, vector<MapPoint*> &vpMapPoints):
      mnFrameId(0), mTimeStamp(kfInit.timeStampNum), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
      mfGridElementWidthInv(kfInit.gridElementWidthInv), mfGridElementHeightInv(kfInit.gridElementHeightInv),
      mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
      mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
      fx(kfInit.fbx), fy(kfInit.fby), cx(kfInit.x), cy(kfInit.y), invfx(kfInit.fx),
      invfy(kfInit.fy), mbf(kfInit.bf), mb(kfInit.b), mThDepth(kfInit.thDepth), N(kfInit.N),
      mvKeys(kfInit.stdvKeyPoint), mvKeysUn(kfInit.stdvKeyPointUn), mvuRight(kfInit.stdvRight), mvDepth(kfInit.stdvDepth),
      mDescriptors(kfInit.matDescriptors.clone()), mBowVec(kfInit.BowVec), mFeatVec(kfInit.FeatVec),
      mnScaleLevels(kfInit.scaleLevels), mfScaleFactor(kfInit.scaleFactor), mfLogScaleFactor(kfInit.logScaleFactor),
      mvScaleFactors(kfInit.scaleFactors), mvLevelSigma2(kfInit.levelSigma2),mvInvLevelSigma2(kfInit.invLevelSigma2),
      mnMinX(kfInit.minX), mnMinY(kfInit.minY), mnMaxX(kfInit.maxX), mnMaxY(kfInit.maxY), mK(kfInit.K),
      mvpMapPoints(vpMapPoints), mpKeyFrameDB(kfDB), mpORBvocabulary(kfInit.vocabulary),
      mbFirstConnection(true), mpParent(NULL), mbNotErase(false), mbToBeErased(false), mbBad(false),
      mHalfBaseline(kfInit.b/2), mpMap(map)
{
    mnId = nNextId++;
}

 
    
    
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}

/**
 * @brief Bag of Words Representation
 *
 * Calculate mBowVec, and disperse the descriptors on the fourth layer, that is, mFeatVec records ni descriptors belonging to the ith node
 * @see ProcessNewKeyFrame()
 */
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    // center is the coordinate of the center of the stereo camera under the camera coordinate system (left eye)
    // The difference between the center point coordinates of the stereo camera and the left eye camera coordinates is only mHalfBaseline on the x-axis,
    // Therefore, it can be seen that the connection between the two cameras in the stereo camera is the x-axis, and the positive direction is that the left eye camera points to the right eye camera
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    // In the world coordinate system, the vector from the center of the left eye camera to the center of the stereo camera, the direction is from the left eye camera to the center of the stereo camera
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

/**
 * @brief Add connections between keyframes
 * 
 * update mConnectedKeyFrameWeights
 * @param pKF    Keyframe
 * @param weight Weight, the number of 3d points observed in this keyframe together with pKF
 */
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        // std::map::count function can only return 0 or 1
        if(!mConnectedKeyFrameWeights.count(pKF))// The count function returns 0, there is no pKF in mConnectedKeyFrameWeights, there is no connection before
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)// The weights of the previous connections are different
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

/**
 * @brief Sort connected keyframes by weight
 * 
 * The updated variables are stored in mvpOrderedConnectedKeyFrames and mvOrderedWeights
 */
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    // http://stackoverflow.com/questions/3389648/difference-between-stdliststdpair-and-stdmap-in-c-stl
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    // Take out all connected keyframes, the type of mConnectedKeyFrameWeights is std::map<KeyFrame*,int>, and the vPairs variable puts the 3D points of the common view in the front, which is convenient for sorting
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    // sort by weight
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;// keyframe
    list<int> lWs;// weight
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    // weights from large to small
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

/**
 * @brief get the keyframe connected to this keyframe
 * @return connected keyframes
 */
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

/**
 * @brief Get the keyframes connected to this keyframe (sorted by weight)
 * @return connected keyframes
 */
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

/**
 * @brief Get the first N keyframes connected to this keyframe (sorted by weight)
 * 
 * If there are fewer than N connected keyframes, return all connected keyframes
 * @param N Top N
 * @return connected keyframes
 */
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

/**
 * @brief Get a keyframe with a weight greater than or equal to w connected to the keyframe
 * @param w Weights
 * @return connected keyframes
 */
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    // http://www.cplusplus.com/reference/algorithm/upper_bound/
    // Find the first iterator greater than w from mvOrderedWeights
    // lower_bound should be used here, because lower_bound returns less than or equal to, and upper_bound can only return the first greater than
    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

/**
 * @brief Get the weight of the key frame and pKF
 * @param  pKF Keyframe
 * @return     Weights
 */
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

/**
 * @brief Add MapPoint to KeyFrame
 * @param pMP MapPoint
 * @param idx Index of MapPoint in KeyFrame
 */
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

/**
 * @brief In keyframes, the number of MapPoints greater than or equal to minObs
 * minObs is a threshold, greater than minObs indicates that the MapPoint is a high-quality MapPoint
 * A high-quality MapPoint will be observed by multiple KeyFrames,
 * @param  minObs Minimum observation
 */
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    // The MapPoint is a high-quality MapPoint
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

/**
 * @brief Get MapPoint Matches
 *
 * Get the MapPoints of this keyframe
 */
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

/**
 * @brief Update graph connections
 * 
 * 1. First obtain all MapPoint points of the key frame, and count the degree of common vision between each key of these 3d points and all other key frames.
 * For each found keyframe, an edge is established, and the weight of the edge is the number of common 3d points between the keyframe and the current keyframe.
 * 2. And the weight must be greater than a threshold. If there is no weight exceeding the threshold, then only the edge with the largest weight is retained (the degree of common vision with other keyframes is relatively high)
 * 3. Sort these connections in descending order of weight to facilitate future processing
 * After updating the covisibility graph, if it has not been initialized, it is initialized to the edge with the largest connection weight (the key frame with the highest degree of common view with other key frames), similar to the maximum spanning tree
 */
void KeyFrame::UpdateConnections()
{
    // Before this function is executed, the key frame only has a connection relationship with MapPoints, this function can update the connection relationship between key frames
    
    //===============1==================================
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        // Get all 3D points for this keyframe
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    // Through the indirect statistics of 3D points, the degree of common vision between all key frames of these 3D points can be observed
    // That is to count how many key frames have a common view relationship with each key frame, and the statistical results are placed in KFcounter
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        // For each MapPoint, observations record all keyframes that can observe the MapPoint
        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            // remove self, self and self are not considered in common
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //===============2==================================
    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    // vPairs records keyframes with other keyframes that share a frame number greater than th
    // pair<int,KeyFrame*>Write the weight of the key frame in the front, and write the key frame in the back for easy sorting later
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            // Find the key frame with the largest corresponding weight (the key frame with the highest degree of common vision)
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            // The corresponding weight needs to be greater than the threshold, and establish a connection to these key frames
            vPairs.push_back(make_pair(mit->second,mit->first));
            // Update mConnectedKeyFrameWeights of this key frame in KFcounter
            // Update the mConnectedKeyFrameWeights of other KeyFrames, and update the connection weights between other keyframes and the current frame
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    // If there is no weight that exceeds the threshold, establish a connection to the keyframe with the largest weight
    if(vPairs.empty())
    {
        // If each keyframe has fewer than th keyframes in view with it,
        // Then only update the mConnectedKeyFrameWeights of the keyframe with the highest degree of common view with other keyframes
        // This is a patch to the previous th threshold that may have been too high
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    // In vPairs, there are key frames and common view weights with high mutual view degree, from large to small
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    //===============3==================================
    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        // Update the connections (weights) of the graph
        mConnectedKeyFrameWeights = KFcounter;//Update the mConnectedKeyFrameWeights of the KeyFrame, and update the connection weight between the current frame and other keyframes
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        // Update spanning tree connections
        if(mbFirstConnection && mnId!=0)
        {
            // Initialize the parent keyframe of this keyframe to the keyframe with the highest degree of common vision
            mpParent = mvpOrderedConnectedKeyFrames.front();
            // Create a two-way connection
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    // Should this place be: (!mbToBeErased), (wubo???)
    // SetBadFlag function is to set mbToBeErased to true, mbToBeErased means that the KeyFrame is erased
    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)// mbNotErase indicates that the KeyFrame should not be erased, so set mbToBeErased to true, indicating that it has been erased, but not actually erased
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);// Let other KeyFrame delete the connection with itself

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);// Let the MapPoint associated with itself delete the connection with itself
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        //Clear the connection between itself and other keyframes
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        // If this keyframe has its own child keyframes, tell these child keyframes that their parent keyframes are dead, and quickly find a new parent keyframe
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;
            
            // Iterate over each child keyframe and let them update the parent keyframe they point to
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                // Sub keyframes traverse each keyframe connected to it (common view keyframes)
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        // If there is a connection relationship (common view) between the child node of the frame and the parent node (grandchild node)
                        // Example: B-->A (B's parent node is A) C-->B (C's parent node is B) D--C (D is connected to C) E--C (E is connected to C) F--C (F is connected to C) D-->A (D's parent node is A) E-->A (E's parent node is A)
                        // Now B hangs up, so C finds the D, E, and F nodes connected to itself that the parent node points to A's D
                        // This process is to find the node that can replace B.
                        // In the above example, B is the current keyframe to be set to SetBadFlag
                        // A is spcit, which is sParentCandidates
                        // C is pKF, pC, which is one of mspChildrens
                        // D, E, F are variables in vpConnected. Since the weight between C and D is larger than the weight between C and E, D is pP
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                // Because the parent node is dead, and the child node finds a new parent node, the child node updates its own parent node
                pC->ChangeParent(pP);
                // Because the child node finds a new parent node and updates the parent node, then the child node is upgraded as an alternate parent node for other child nodes
                sParentCandidates.insert(pC);
                // The child node is processed
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        // If there are still child nodes and no new parent node is found
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                // Directly use the parent node of the parent node as its own parent node
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

// r is the side length (radius)
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // floor is rounded down, mfGridElementWidthInv is how many grids each pixel occupies
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    // ceil round up
    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i i-th keypoint
 * @return   3D point (relative to the world coordinate system)
 */
cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        // Backproject from the 2D image to the camera coordinate system
        // mvDepth is obtained in the ComputeStereoMatches function
        // The feature point before correction corresponding to mvDepth, so here is the back projection of the feature point before correction
        // But in Frame::UnprojectStereo, it is the back-projection of the corrected feature point mvKeysUn
        // In the ComputeStereoMatches function, the depth of the corrected feature points should be calculated
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        // Convert from camera coordinate system to world coordinate system
        // Twc is the transformation matrix from the camera coordinate system to the world coordinate system
        // Twc.rosRange(0,3).colRange(0,3) takes the first 3 rows and first 3 columns of the Twc matrix
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

/**
 * @brief Evaluate the current keyframe scene depth, q=2 means the median
 * @param q q=2
 * @return Median Depth
 */
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;// The third line of (R*x3Dw+t), which is z
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

// map serialization addition
// Default serializing Constructor
KeyFrame::KeyFrame():
    mnFrameId(0),  mTimeStamp(0.0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(0.0), mfGridElementHeightInv(0.0),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(0.0), fy(0.0), cx(0.0), cy(0.0), invfx(0.0), invfy(0.0),
    mbf(0.0), mb(0.0), mThDepth(0.0), N(0), mnScaleLevels(0), mfScaleFactor(0),
    mfLogScaleFactor(0.0),
    mnMinX(0), mnMinY(0), mnMaxX(0),
    mnMaxY(0)
{}
template<class Archive>
void KeyFrame::serialize(Archive &ar, const unsigned int version)
{
    // no mutex needed vars
    ar & nNextId;
    ar & mnId;
    ar & const_cast<long unsigned int &>(mnFrameId);
    ar & const_cast<double &>(mTimeStamp);
    // Grid related vars
    ar & const_cast<int &>(mnGridCols);
    ar & const_cast<int &>(mnGridRows);
    ar & const_cast<float &>(mfGridElementWidthInv);
    ar & const_cast<float &>(mfGridElementHeightInv);
    // Tracking related vars
    ar & mnTrackReferenceForFrame & mnFuseTargetForKF;
    // LocalMaping related vars
    ar & mnBALocalForKF & mnBAFixedForKF;
    // KeyFrameDB related vars
    ar & mnLoopQuery & mnLoopWords & mLoopScore & mnRelocQuery & mnRelocWords & mRelocScore;
    // LoopClosing related vars
    ar & mTcwGBA & mTcwBefGBA & mnBAGlobalForKF;
    // calibration parameters
    ar & const_cast<float &>(fx) & const_cast<float &>(fy) & const_cast<float &>(cx) & const_cast<float &>(cy);
    ar & const_cast<float &>(invfx) & const_cast<float &>(invfy) & const_cast<float &>(mbf);
    ar & const_cast<float &>(mb) & const_cast<float &>(mThDepth);
    // Number of KeyPoints;
    ar & const_cast<int &>(N);
    // KeyPoints, stereo coordinate and descriptors
    ar & const_cast<std::vector<cv::KeyPoint> &>(mvKeys);
    ar & const_cast<std::vector<cv::KeyPoint> &>(mvKeysUn);
    ar & const_cast<std::vector<float> &>(mvuRight);
    ar & const_cast<std::vector<float> &>(mvDepth);
    ar & const_cast<cv::Mat &>(mDescriptors);
    // Bow
    ar & mBowVec & mFeatVec;
    // Pose relative to parent
    ar & mTcp;
    // Scale related
    ar & const_cast<int &>(mnScaleLevels) & const_cast<float &>(mfScaleFactor) & const_cast<float &>(mfLogScaleFactor);
    ar & const_cast<std::vector<float> &>(mvScaleFactors) & const_cast<std::vector<float> &>(mvLevelSigma2) & const_cast<std::vector<float> &>(mvInvLevelSigma2);
    // Image bounds and calibration
    ar & const_cast<int &>(mnMinX) & const_cast<int &>(mnMinY) & const_cast<int &>(mnMaxX) & const_cast<int &>(mnMaxY);
    ar & const_cast<cv::Mat &>(mK);

    // mutex needed vars, but don't lock mutex in the save/load procedure
    {
        unique_lock<mutex> lock_pose(mMutexPose);
        ar & Tcw & Twc & Ow & Cw;
    }
    {
        unique_lock<mutex> lock_feature(mMutexFeatures);
        ar & mvpMapPoints; // hope boost deal with the pointer graph well
    }
    // BoW
    ar & mpKeyFrameDB;
    // mpORBvocabulary restore elsewhere(see SetORBvocab)
    {
        // Grid related
        unique_lock<mutex> lock_connection(mMutexConnections);
        ar & mGrid & mConnectedKeyFrameWeights & mvpOrderedConnectedKeyFrames & mvOrderedWeights;
        // Spanning Tree and Loop Edges
        ar & mbFirstConnection & mpParent & mspChildrens & mspLoopEdges;
        // Bad flags
        ar & mbNotErase & mbToBeErased & mbBad & mHalfBaseline;
    }
    // Map Points
    ar & mpMap;
    // don't save mutex
}
template void KeyFrame::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void KeyFrame::serialize(boost::archive::binary_oarchive&, const unsigned int);

} //namespace ORB_SLAM
