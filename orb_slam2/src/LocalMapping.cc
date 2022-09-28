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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        // Tell Tracking that LocalMapping is busy,
        // The key frames processed by the LocalMapping thread are all sent by the Tracking thread
        // It is best not to send the Tracking thread too fast until the LocalMapping thread has not finished processing the key frame
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        // The list of keyframes waiting to be processed is not empty
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            // VI-A keyframe insertion
            // Calculate the BoW map of the keyframe feature points and insert the keyframe into the map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            // VI-B recent map points culling
            // Eliminate unqualified MapPoints introduced in ProcessNewKeyFrame function
            MapPointCulling();

            // Triangulate new MapPoints
            // VI-C new map points creation
            // Some MapPoints are recovered by triangulation with adjacent keyframes during camera motion
            CreateNewMapPoints();

            // The last keyframe in the queue has been processed
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                // Check and fuse the MapPoints duplicated between the current keyframe and adjacent frames (two levels adjacent)
                SearchInNeighbors();
            }

            mbAbortBA = false;

            // The last key frame in the queue has been processed, and the closed loop detection did not request to stop LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                // VI-E local keyframes culling
                // Detect and remove redundant keyframes in adjacent keyframes of the current frame
                // The criterion for culling is: 90% of the MapPoints of the keyframe can be observed by other keyframes
                // trick!
                // In Tracking, the key frame is first handed over to the LocalMapping thread
                // And the conditions of the InsertKeyFrame function in Tracking are relatively loose, and the key frames handed to the LocalMapping thread will be denser
                // delete redundant keyframes here
                KeyFrameCulling();
            }

            // Add the current frame to the closed loop detection queue
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                std::this_thread::sleep_for(std::chrono::microseconds(3000));
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }

    SetFinish();
}

/**
 * @brief Insert keyframes
 *
 * Insert keyframes into the map for future local map optimization
 * Here is just inserting keyframes into the list and waiting
 * @param pKF KeyFrame
 */
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    // insert keyframes into the list
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


/**
 * @brief See if there are keyframes waiting to be inserted in the list
 * @return Returns true if it exists
 */
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief Process keyframes in a list
 * 
 * - Calculate Bow, speed up triangulation of new MapPoints
 * - Associate the current keyframe to MapPoints, and update the average observation direction and observation distance range of MapPoints
 * - Insert keyframes, update Covisibility graph and Essential graph
 * @see VI-A keyframe insertion
 */
void LocalMapping::ProcessNewKeyFrame()
{
    // Step 1: Take a frame of keyframes from the buffer queue
    // Tracking thread inserts keyframes into LocalMapping and stores them in the queue
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // Get a keyframe waiting to be inserted from the list
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    // Step 2: Calculate the Bow mapping relationship of the key frame feature points
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // Step 3: Bind the MapPoints on the new match to the current keyframe in the process of tracking the local map
    // Match the MapPoints in the local map with the current frame in the TrackLocalMap function,
    // but do not associate these matching MapPoints with the current frame
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // MapPoints generated by non-current frame
                // Update properties for the MapPoints tracked by the current frame during the tracking process
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    // add observations
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    // Get the average observation direction and observation distance range of the point
                    pMP->UpdateNormalAndDepth();
                    // After adding keyframes, update the best descriptor of the 3d point
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    // MapPoints generated by the current frame
                    // Put the newly inserted MapPoints in the binocular or RGBD tracking process into mlpRecentAddedMapPoints, waiting to be checked
                    // MapPoints are also generated by triangulation in the CreateNewMapPoints function
                    // These MapPoints will be checked by the MapPointCulling function
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    // Step 4: Update the connection between keyframes, Covisibility graph and Essential graph (tree)
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    // Step 5: Insert the keyframe into the map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief Eliminate poor quality MapPoints introduced in ProcessNewKeyFrame and CreateNewMapPoints functions
 * @see VI-B recent map points culling
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    // Traverse the MapPoints waiting to be checked
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            // Step 1: MapPoints that are already dead points are deleted directly from the check list
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            // Step 2: Eliminate MapPoints that do not meet VI-B conditions
            // VI-B Condition 1:
            // The ratio of the number of Frames tracked to the MapPoint should be greater than 25% compared to the number of Frames that are expected to observe the MapPoint
            // IncreaseFound / IncreaseVisible < 25%, note that it is not necessarily a key frame.
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            // Step 3: Eliminate MapPoints that do not meet VI-B conditions
            // VI-B Condition 2: No less than 2 key frames have passed since the establishment of this point
            // But the number of key frames observed at this point does not exceed cnThObs frames, then the point inspection fails
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            // Step 4: Starting from the establishment of this point, 3 key frames have passed without being eliminated, and it is considered to be a high-quality point
            // So there is no SetBadFlag(), just delete from the queue, give up and continue to detect the MapPoint
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

/**
 * Some MapPoints are recovered by triangulation during camera movement and key frames with a high degree of common vision
 */
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    // Step 1: Find the nn adjacent frames vpNeighKFs with the highest common view degree in the common view key frames of the current key frame
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    // Get the coordinates of the current keyframe in the world coordinate system
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    // Step 2: Traverse adjacent keyframes vpNeighKFs
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // Coordinates of adjacent keyframes in world coordinates
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // Baseline vector, camera displacement between two keyframes
        cv::Mat vBaseline = Ow2-Ow1;
        // baseline length
        const float baseline = cv::norm(vBaseline);

        // Step 3: Determine whether the baseline of camera motion is long enough
        if(!mbMonocular)
        {
            // If it is a stereo camera, if the keyframe spacing is too small, no 3D points will be generated
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            // Median scene depth of adjacent keyframes
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // ratio of baseline and depth of field
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            
            // If it is very far (the scale is very small), then the current adjacent keyframes are not considered, and 3D points are not generated
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        // Step 4: Calculate the fundamental matrix between the two keyframes based on their poses
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        // Step 5: Limit the search range during matching through epipolar constraints, and perform feature point matching
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // Step 6: Generate 3D points by triangulation for each pair of matches, the Triangulate function is similar
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            // Step 6.1: Take out matching feature points

            // The index of the current matching pair in the current keyframe
            const int &idx1 = vMatchedIndices[ikp].first;
            
            // The index of the current matching pair in the adjacent keyframe
            const int &idx2 = vMatchedIndices[ikp].second;

            // Currently matching feature points in the current keyframe
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            // mvuRight stores the binocular depth value, if it is not binocular, its value will be -1
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            // Feature points currently matched in adjacent keyframes
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            // mvuRight stores the binocular depth value, if it is not binocular, its value will be -1
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            // Step 6.2: Use the matching point back projection to get the parallax angle
            // feature point back projection
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            // From the camera coordinate system to the world coordinate system, get the cosine value of the parallax angle
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            // Add 1 to make cosParallaxStereo randomly initialized to a large value
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            // Step 6.3: For binoculars, use binoculars to get the parallax angle
            if(bStereo1)// binocular, and has depth
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)// binocular, and has depth
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            // Get the parallax angle of binocular observation
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            // Step 6.4: Triangulate to restore 3D points
            cv::Mat x3D;
            // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998) indicates that the parallax angle is normal
            // cosParallaxRays<cosParallaxStereo indicates that the parallax angle is small
            // Use trigonometry to restore 3D points when the parallax angle is small, and use binocular to restore 3D points when the parallax angle is large (binocular and depth are valid
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                // See Triangulate function in Initializer.cpp
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            // Step 6.5: Check if the generated 3D point is in front of the camera
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            // Step 6.6: Calculate the reprojection error of the 3D point under the current keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                // Threshold calculated based on chi-square test (assuming the measurement has a one-pixel deviation)
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            // Calculate the reprojection error of the 3D point under another keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                // Threshold calculated based on the chi-square test (assuming the measurement has a one-pixel deviation)
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            // Step 6.7: Check for scale continuity

            // In the world coordinate system, the vector between the 3D point and the camera, the direction is from the camera to the 3D point
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            // ratioDist is the distance ratio without considering the pyramid scale
            const float ratioDist = dist2/dist1;
            // scale of the pyramid scale factor
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            // ratioDist*ratioFactor < ratioOctave or ratioDist/ratioOctave > ratioFactor indicates that scale changes are continuous
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            // Step 6.8: The 3D point is successfully generated by triangulation, and it is constructed as a MapPoint
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            // Step 6.9: Add properties to this MapPoint:
            // a. Observe the key frame of the MapPoint
            // b. The descriptor of the MapPoint
            // c. The average observation direction and depth range of the MapPoint
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            // Step 6.8: Put the newly generated point into the detection queue
            // These MapPoints will be checked by the MapPointCulling function
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

/**
 * Check and fuse MapPoints that duplicate the current keyframe and adjacent frames (two levels adjacent)
 */
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // Step 1: Obtain the adjacent keyframes with the top nn weights of the current keyframe in the covisibility graph
    // Find the first-level adjacent and second-level adjacent keyframes of the current frame
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);// Add one level adjacent frame
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;// and mark already added

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);// store the second-level adjacent frame
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    
    // Step 2: Integrate the MapPoints of the current frame with the first-level and second-level adjacent frames (MapPoints)
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        // Project the MapPoints of the current frame to the adjacent key frame pKFi, and determine whether there are duplicate MapPoints
        // 1. If the MapPoint can match the feature point of the key frame, and the point has a corresponding MapPoint, then merge the two MapPoints (select the one with more observations)
        // 2. If the MapPoint can match the feature point of the key frame, and the point does not have a corresponding MapPoint, then add a MapPoint to the point
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    // A collection of all MapPoints used to store first-level adjacency and second-level adjacency keyframes
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    // Step 3: Integrate the MapPoints of the first-level and second-level adjacent frames with the current frame (MapPoints)
    // Traverse each first-level adjacency and second-level adjacency keyframes
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        // Traverse all MapPoints in the current first-level adjacency and second-level adjacency keyframes
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            
            // Determine whether MapPoints is a dead point, or whether it has been added to the collection vpFuseCandidates
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            
            // Join the collection and mark it as already joined
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    // Step 4: Update the descriptor, depth, observation main direction and other attributes of the current frame MapPoints
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // Get the best descriptor among all keyframes where pMP is found
                pMP->ComputeDistinctiveDescriptors();
                
                // Update the average observation direction and observation distance
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    // Step 5: Update the connection relationship with other frames after updating the MapPoints of the current frame
    // Update the covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

/**
 * Calculate the fundamental matrix between two keyframes based on the poses of the two keyframes
 * @param  pKF1 keyframe 1
 * @param  pKF2 keyframe 2
 * @return      fundamental matrix
 */
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    // Essential Matrix: t12 fork multiplied by R12
    // Fundamental Matrix: inv(K1)*E*inv(K2)
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

    
/**
 * @brief keyframe culling
 * 
 * For a keyframe in the Covisibility Graph, if more than 90% of the MapPoints can be observed by other keyframes (at least 3), the keyframe is considered redundant.
 * @see VI-E Local Keyframe Culling
 */
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    // Step 1: Extract the co-view keyframe of the current frame according to the Covisibility Graph
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    // Iterate over all local keyframes
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        // Step 2: Extract MapPoints for each common view keyframe
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        // Step 3: Traverse the MapPoints of the local keyframe to determine whether more than 90% of the MapPoints can be observed by other keyframes (at least 3)
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        // For binocular, only consider near MapPoints, no more than mbf * 35 / fx
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    // MapPoints are observed by at least three keyframes
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        // Determine whether the MapPoint is observed by three keyframes at the same time
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            // Scale Condition 
                            // The scale constraint requires that the feature scale of MapPoint in this local key frame is greater than (or similar to) the feature scale of other key frames
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                // Three key frames of the same scale have been found to observe the MapPoint, so don't continue to look for it
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        // This MapPoint is observed by at least three keyframes
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        // Step 4: If more than 90% of the MapPoints of the local key frame can be observed by other key frames (at least 3), it is considered as redundant key frame
        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
