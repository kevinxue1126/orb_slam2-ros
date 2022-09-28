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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

// If the first letter of the variable name in the program is "m", it means a member variable in the class, member
// first, second letter:
// "p" indicates the pointer data type
// "n" means int type
// "b" means bool type
// "s" means set type
// "v" represents the vector data type
// 'l' means list data type
// "KF" represents the KeyPoint data type

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys),
    mpFrameDrawer(pFrameDrawer), mpMap(pMap), mnLastRelocFrameId(0), mnMinimumKeyFrames(5)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // image correction factor
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // binocular camera baseline * fx 50
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    // The number of feature points extracted per frame is 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // Change the scale of the image when building the pyramid 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // The level of the scale pyramid is 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // The default threshold for extracting fast feature points is 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // If the default threshold does not extract enough fast feature points, use the minimum threshold of 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // The tracking process will use mpORBextractorLeft as the feature point extractor
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // If it is binocular, mpORBextractorRight will also be used as the right-eye feature point extractor during the tracking process
    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // During monocular initialization, mpIniORBextractor will be used as the feature point extractor
    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        // Determine the far/near threshold of a 3D point mbf * 35 / fx
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        // Factor when depth camera disparity is converted to depth
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}


// Input left and right eye images, which can be RGB, BGR, RGBA, GRAY
// 1. Convert the image to mImGray and imGrayRight and initialize mCurrentFrame
// 2. Perform the tracking process
// Output the transformation matrix from the world coordinate system to the camera coordinate system of the frame
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    // Step 1: Convert RGB or RGBA image to grayscale
    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    // Step 2: Construct Frame
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // Step 3: Trace
    Track();

    return mCurrentFrame.mTcw.clone();
}


// Input left eye RGB or RGBA image and depth map
// 1. Convert the image to mImGray and imDepth and initialize mCurrentFrame
// 2. Perform the tracking process
// Output the transformation matrix from the world coordinate system to the camera coordinate system of the frame
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    // Step 1: Convert RGB or RGBA image to grayscale
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // Step 2: Convert the disparity of the depth camera to Depth
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    // Step 3: Construct Frame
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // Step 4: Trace
    Track();

    return mCurrentFrame.mTcw.clone();
}


// Input left eye RGB or RGBA image
// 1. Convert the image to mImGray and initialize mCurrentFrame
// 2. Perform the tracking process
// Output the transformation matrix from the world coordinate system to the camera coordinate system of the frame
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    // Step 1: Convert RGB or RGBA image to grayscale
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // Step 2: Construct Frame
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // Step 3: Trace
    Track();

    return mCurrentFrame.mTcw.clone();
}

/**
 * @brief Main tracking function. It is independent of the input sensor.
 *
 * Tracking thread
 */
void Tracking::Track()
{
    // track contains two parts: estimating motion, tracking local map
    
    // mState is the tracking state machine
    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // If the image has been reset, or is running for the first time, it is NO_IMAGE_YET state
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // mLastProcessedState stores the latest state of Tracking for drawing in FrameDrawer
    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // Step 1: Initialize
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else// Step 2: Trace
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                // Step 2.1: Track the previous frame or reference frame or relocate

                // The kinematic model is empty or just relocated
                // mCurrentFrame.mnId<mnLastRelocFrameId+2 This judgment should not have
                // TrackWithMotionModel should be preferred as long as mVlocity is not empty
                // mnLastRelocFrameId the last relocated frame
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // Use the pose of the previous frame as the initial pose of the current frame
                    // Find the matching point of the current frame feature point in the reference frame by BoW
                    // Optimize each feature point to correspond to the 3D point reprojection error to get the pose
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    // Set the initial pose of the current frame according to the constant speed model
                    // Find the matching point of the feature point of the current frame in the reference frame by projection
                    // Optimize the projection error of the 3D point corresponding to each feature point to get the pose
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        // TrackReferenceKeyFrame is a tracking reference frame. It cannot predict the pose of the current frame according to the fixed motion speed model, and accelerate the matching through bow (SearchByBow)
                        // Finally, the optimized pose is obtained by optimization
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                // BOW search, PnP solves the pose
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            // only tracking, local map does not work
 
            // Step 2.1: Track the previous frame or reference frame or relocate

            // tracking is lost
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                // mbVO is a variable only when mbOnlyTracking is true
                // If mbVO is false, it means that this frame matches a lot of MapPoints, and the tracking is normal.
                // mbVO is true to indicate that this frame matches very few MapPoints, less than 10, the rhythm to kneel
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    // relocation did not succeed, but if tracking succeeded
                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            // Isn't this code a bit redundant? It should be put into the TrackLocalMap function to do it uniformly
                            // Update the observed degree of MapPoints of the current frame
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)// As long as the relocation is successful, the entire tracking process will proceed normally (location and tracking, more believe in relocation)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // use the latest keyframe as the reference frame
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // Step 2.2: After the initial pose is obtained by matching between frames, now track the local map to obtain more matches, and optimize the current pose
        // local map: the current frame, the current frame's MapPoints, the current key frame and other key frames co-viewing relationship
        // In step 2.1, it is mainly two-by-two tracking (the constant-speed model tracks the previous frame and tracks the reference frame). Here, after searching for local key frames, collect all local MapPoints,
        // Then perform projection matching between the local MapPoints and the current frame, and then perform Pose optimization after getting more matching MapPoints
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                // Step 2.3: Update the mVelocity in the constant velocity motion model TrackWithMotionModel
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            //mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            // Step 2.4: Clear the MapPoints temporarily added for the current frame in UpdateLastFrame
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    // Exclude MapPoints added for tracking in UpdateLastFrame function
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            // Step 2.5: Clear the temporary MapPoints, which are generated in the UpdateLastFrame function of TrackWithMotionModel (only binocular and rgbd)
            // In step 2.4, these MapPoints are only removed from the current frame, here they are deleted from the MapPoints database
            // What is generated here is only to improve the inter-frame tracking effect of the binocular or rgbd camera, and it will be thrown away after it is used up, and it will not be added to the map
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            // This is not only to clear mlpTemporalPoints, but also to delete the MapPoint pointed to by the pointer through delete pMP
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            // Step 2.6: Detect and insert keyframes, which will generate new MapPoints for binoculars
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=mnMinimumKeyFrames)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // save the data of the previous frame
        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // Step 3: Record pose information for trajectory reproduction
    if(!mCurrentFrame.mTcw.empty())
    {
        // Calculate relative pose T_currentFrame_referenceKeyFrame
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


/**
 * @brief binocular and rgbd map initialization, since stereo has a depth map, it can be initialized in a single frame
 *
 * Directly generate MapPoints due to depth information
 */
void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        // Step 1: Set the initial pose
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        // Step 2: Construct the current frame as the initial keyframe
        // The data type of mCurrentFrame is Frame
        // KeyFrame contains Frame, map 3D points, and BoW
        // There is an mpMap in KeyFrame, an mpMap in Tracking, and the mpMap in KeyFrame all point to this mpMap in Tracking
        // There is an mpKeyFrameDB in KeyFrame, an mpKeyFrameDB in Tracking, and mpMap in KeyFrame all point to this mpKeyFrameDB in Tracking
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // ! ! ! Is there a missing pKFini->ComputeBoW();

        // Insert KeyFrame in the map
        // The KeyFrame contains the map, and in turn the map also contains the KeyFrame, which contains each other
        // Step 3: Add this initial keyframe to the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        // Step 4: Construct MapPoint for each feature point by stereo depth
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                // Step 4.1: Obtain the 3D coordinates of the feature point through back projection
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                // Step 4.2: Construct the 3D point as a MapPoint
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                
                // Step 4.3: Add properties to this MapPoint:
                // a. Observe the key frame of the MapPoint
                // b. The descriptor of the MapPoint
                // c. The average observation direction and depth range of the MapPoint

                // a. Indicates which feature point of which KeyFrame the MapPoint can be observed by
                pNewMP->AddObservation(pKFini,i);
                // b. Select the descriptor with the highest distinguishing read from the feature points that observe the MapPoint
                pNewMP->ComputeDistinctiveDescriptors();
                // c. Update the average observation direction of the MapPoint and the range of the observation distance
                pNewMP->UpdateNormalAndDepth();
                // Step 4.4: Add the MapPoint to the map
                mpMap->AddMapPoint(pNewMP);
                // Step 4.5: Indicate which feature point of the KeyFrame can observe which 3D point
                pKFini->AddMapPoint(pNewMP,i);
                
                
                // Step 4.6: Add this MapPoint to mvpMapPoints of the current frame
                // Create an index between the feature point of the current Frame and the MapPoint
                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // Step 5: Add this initial keyframe to the local map
        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        // use the current (latest) local MapPoints as ReferenceMapPoints
        // ReferenceMapPoints is used when the DrawMapPoints function draws
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        //mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

/**
 * @brief single destination graph initialization
 *
 * Calculate the fundamental matrix and the homography matrix in parallel, select one of the models, and recover the relative pose and point cloud between the first two frames
 * Get the matching, relative motion, and initial MapPoints of the initial two frames
 */
void Tracking::MonocularInitialization()
{

    // If the monocular initializer has not been created yet, create the monocular initializer
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            // Step 1: Get the first frame for initialization, two frames are needed for initialization
            mInitialFrame = Frame(mCurrentFrame);
            // record the most recent frame
            mLastFrame = Frame(mCurrentFrame);
            // The biggest case of mvbPrevMatched is that all feature points are tracked
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            // Construct initializer from current frame sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        // Step 2: If the number of feature points in the current frame is greater than 100, get the second frame for monocular initialization
        // If there are too few feature points in the current frame, rebuild the initializer
        // Therefore, the initialization process can be continued only when the number of feature points in two consecutive frames is greater than 100.
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        // Step 3: Find matching feature point pairs in mInitialFrame and mCurrentFrame
        // mvbPrevMatched is the feature point of the previous frame, which stores which points in mInitialFrame will be matched next
        // mvIniMatches stores the matching feature points between mInitialFrame and mCurrentFrame
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // Step 4: If there are too few matching points between the initialized two frames, reinitialize
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // Step 5: Perform monocular initialization through H model or F model to obtain relative motion between two frames, initial MapPoints
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            // Step 6: Remove those matching points that cannot be triangulated
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            // Construct Tcw from Rcw and tcw, and assign it to mTcw, mTcw is the transformation matrix from the world coordinate system to the frame
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // Step 6: Pack the triangulated 3D points into MapPoints
            // The Initialize function will get mvIniP3D,
            // mvIniP3D is a container of type cv::Point3f, which is a temporary variable that stores 3D points.
            // CreateInitialMapMonocular wraps 3D points into MapPoint types and stores them in KeyFrame and Map
            CreateInitialMapMonocular();
        }
    }
}

/**
 * @brief CreateInitialMapMonocular
 *
 * Generate MapPoints for monocular camera triangulation
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    // Step 1: Convert the descriptor of the initial keyframe to BoW
    pKFini->ComputeBoW();
    // Step 2: Convert the descriptor of the current keyframe to BoW
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // Step 3: Insert keyframes into the map
    // All keyframes must be inserted into the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    // Step 4: Wrap 3D Points into MapPoints
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        // Step 4.1: Construct MapPoint with 3D Points
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        // Step 4.2: Add properties to this MapPoint:
        // a. Observe the key frame of the MapPoint
        // b. The descriptor of the MapPoint
        // c. The average observation direction and depth range of the MapPoint

        // Step 4.3: Indicate which feature point of the KeyFrame can observe which 3D point
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // a. Indicates which feature point of which KeyFrame the MapPoint can be observed by
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // b. Select the descriptor with the highest distinguishing read from the feature points that observe the MapPoint
        pMP->ComputeDistinctiveDescriptors();
        // c. Update the average observation direction of the MapPoint and the range of the observation distance
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        // Step 4.4: Add the MapPoint to the map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    // Step 5: Update the connection relationship between key frames, and perform a key connection relationship update for a newly created key frame
    // Establish edges between 3D points and keyframes, each edge has a weight, and the weight of the edge is the number of common 3D points between the keyframe and the current frame
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // Step 5: BA optimization
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    // Step 6:!!! Normalize the median depth of MapPoints to 1, and normalize the transformation between the two frames
    // The monocular sensor cannot restore the true depth, here the median depth of the point cloud (Euclidean distance, not z) is normalized to 1
    // Evaluate keyframe scene depth, q=2 means median
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    // Scale the translation amount according to the normalized scale of the point cloud
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    // This part is similar to SteroInitialization()
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    //mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;// Initialization is successful, so far, the initialization process is complete
}

/**
 * @brief Check if the MapPoints in the previous frame were replaced
 * keyframe has fuse mappoint in local_mapping and loopclosure.
 * Since these mappoints have been changed, and only the mappoints of the key frames are updated, for the normal frames of mLastFrame, also check and update the mappoints
 * @see LocalMapping::SearchInNeighbors()
 */
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


/**
 * @brief tracks MapPoints referenced to keyframes
 *
 * 1. Calculate the word bag of the current frame, and assign the feature points of the current frame to the nodes of a specific layer
 * 2. Match descriptors belonging to the same node
 * 3. Estimate the pose of the current frame based on the matching pair
 * 4. Eliminate false matches based on pose
 * @return true if the number of matches detected by the reprojection error is greater than 10
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // Step 1: Convert the descriptor of the current frame to a BoW vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // Step 2: Speed up feature point matching between current frame and reference frame through BoW of feature points
    // The matching relationship of feature points is maintained by MapPoints
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    // Step 3: Use the pose of the previous frame as the initial value of the pose of the current frame
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);// Set the initial value with the last Tcw, which can converge faster in PoseOptimization

    // Step 4: Obtain the pose by optimizing the 3D-2D reprojection error
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // Step 5: Eliminate the optimized outlier matching points (MapPoints)
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

/**
 * @brief binocular or rgbd camera generates new MapPoints based on the depth value of the previous frame
 *
 * In the case of binocular and rgbd, select some points with a smaller depth (more reliable) \n
 * Can generate some new MapPoints from depth values
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // Step 1: Update the pose of the last frame
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    // If the previous frame is a key frame, or a single purpose case, exit
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Step 2: For binocular or rgbd cameras, temporarily generate new MapPoints for the previous frame
    // Note that these MapPoints are not added to the Map and will be deleted at the end of the tracking
    // During the tracking process, it is necessary to project the MapPoints of the previous frame to the current frame to narrow the matching range and speed up the feature point matching between the current frame and the previous frame.
    
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // Step 2.1: Get the feature points with depth values ​​in the previous frame
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    // Step 2.2: Sort by depth from small to large
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // Step 2.3: Pack the points that are closer into MapPoints
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            // These do not pass after generating MapPoints:
            // a.AddMapPoint,
            // b.AddObservation,
            // c.ComputeDistinctiveDescriptors,
            // d.UpdateNormalAndDepth adds properties,
            // These MapPoints are only to improve the tracking success rate of binocular and RGBD
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;// add new MapPoint

            // MapPoints marked as temporarily added, and then all deleted before CreateNewKeyFrame
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

/**
 * @brief Track the MapPoints of the previous frame according to the uniform velocity model
 *
 * 1. For non-monocular cases, some new MapPoints (temporary) need to be generated for the previous frame
 * 2. Project the MapPoints of the previous frame onto the image plane of the current frame, and perform region matching at the projected position
 * 3. Estimate the pose of the current frame based on the matching pair
 * 4. Eliminate false matches based on pose
 * @return if the number of matches is greater than 10, return true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // Step 1: For binocular or rgbd cameras, generate new MapPoints based on the depth value of the previous keyframe
    // (During the tracking process, it is necessary to match the feature points of the current frame and the previous frame, and project the MapPoints of the previous frame to the current frame to narrow the matching range)
    // During the tracking process, remove the MapPoint of the outlier. If the MapPoint is not added in time, it will gradually decrease
    // The function of this function is to supplement the number of MapPoints in the previous frame of the RGBD and binocular cameras
    UpdateLastFrame();

    // Estimate the pose of the current frame according to the Const Velocity Model (it is considered that the relative motion between the two frames is the same as the relative motion between the previous two frames)
    // mVelocity is the difference between the last and previous frame poses
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    
    // Step 2: Track the MapPoints of the previous frame according to the uniform velocity model
    // Reduce the matching range of feature points according to the position of the 3D point projection corresponding to the feature point of the previous frame
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    // Step 3: Optimize pose, only-pose BA optimization
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // Step 4: After optimizing the pose, remove the outlier's mvpMapPoints
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

/**
 * @brief tracks the MapPoints of the Local Map
 *
 * 1. Update the local map, including local keyframes and keypoints
 * 2. Projection matching for local MapPoints
 * 3. Estimate the pose of the current frame based on the matching pair
 * 4. Eliminate false matches based on pose
 * @return true if success
 * @see V-D track Local Map
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    // Step 1: Update local keyframes mvp Local KeyFrames and local map points mvpLocalMapPoints
    UpdateLocalMap();

    // Step 2: Find MapPoints in the local map that match the current frame
    SearchLocalPoints();

    // Optimize Pose
    // Before this function, there are pose optimizations in Relocalization, Track Reference KeyFrame, Track With MotionModel, 
    // Step 3: After updating all local MapPoints, optimize the pose again
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    // Step 3: Update the observed degree of MapPoints of the current frame, and count the effect of tracking the local map
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            // Since the MapPoints of the current frame can be observed by the current frame, add 1 to the observed statistics
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    // This MapPoint has been observed by other keyframes
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    // Record the MapPoints tracked by the current frame for statistical tracking effects
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // Step 4: Determine whether the tracking is successful
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


/**
 * @brief whether the current frame is a key frame
 * @return true if needed
 */
bool Tracking::NeedNewKeyFrame()
{
    // Step 1: If the user chooses to reposition on the interface, then no keyframes will be inserted
    // Since MapPoint is generated during the insertion of keyframes, the point cloud and keyframes on the map will not increase after the user chooses to relocate
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // Step 2: Determine whether the time since the last keyframe was inserted is too short
    // mCurrentFrame.mnId is the ID of the current frame
    // mnLastRelocFrameId is the ID of the last relocated frame
    // mMaxFrames is equal to the frame rate of the image input
    // If there are few keyframes, consider inserting keyframes
    // or more than 1s from the last relocation, consider inserting keyframes
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // Step 3: Get the number of MapPoints tracked by the reference keyframe
    // In the UpdateLocalKeyFrames function, the key frame with the highest degree of common view with the current key frame will be set as the reference key frame of the current frame
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // Step 4: Query if the local map manager is busy
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // Step 5: For binocular or RGBD cameras, count the total number of MapPoints that can be added and the number of MapPoints tracked into the map
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)// binocular or rgbd
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Step 6: Decide whether to insert keyframes
    //Thresholds
    // Set the inlier threshold, the inlier ratio that matches the feature points of the previous frame
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;// There is only one keyframe, so the threshold for inserting keyframes is very low

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                // Do not block too many keyframes in the queue
                // Tracking insertion of key frames is not directly inserted, but first inserted into mlNewKeyFrames,
                // Then localmapper pops out and inserts them into mspKeyFrames one by one
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

/**
 * @brief creates a new keyframe
 *
 * For non-single-purpose cases, create new MapPoints at the same time
 */
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // Step 1: Construct the current frame into a keyframe
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // Step 2: Set the current keyframe as the reference keyframe for the current frame
    // In the UpdateLocalKeyFrames function, the key frame with the highest degree of common view with the current key frame will be set as the reference key frame of the current frame
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // This code has the same function as the part of the code in UpdateLastFrame
    // Step 3: For binocular or rgbd cameras, generate new MapPoints for the current frame
    if(mSensor!=System::MONOCULAR)
    {
        // Calculate mRcw, mtcw and mRwc, mOw according to Tcw
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        // Step 3.1: Get the feature points whose current frame depth is less than the threshold
        // Create new MapPoint, depth < mThDepth
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            // Step 3.2: Sort by depth from small to large
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // Step 3.3: Pack the closer points into MapPoints
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // These operations of adding properties are done every time a MapPoint is created
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                // This determines the density of the map point cloud for the binocular and rgbd cameras
                // But it's not good to change these just to make the map dense,
                // Because these MapPoints will participate in the entire slam process later
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief tracks Local MapPoints
 *
 * Find the points within the field of view of the current frame in the local map, and perform projection matching between the points within the field of view and the feature points of the current frame
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // Step 1: Traverse the mvpMapPoints of the current frame and mark these MapPoints not to participate in subsequent searches
    // Because the current mvpMapPoints must be in view of the current frame
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // Update the number of frames where the point can be observed plus 1
                pMP->IncreaseVisible();
                // mark the point as being observed by the current frame
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // mark the point not to be projected in the future because it has already been matched
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // Step 2: Project all local MapPoints to the current frame, determine whether they are within the field of view, and then perform projection matching
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        // MapPoint has been observed by the current frame and no longer determines whether it can be observed by the current frame
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // Step 2.1: Determine whether the point in LocalMapPoints is in view
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            // Add 1 to the number of frames where the point is observed, and the MapPoint is within the field of view of some frames
            pMP->IncreaseVisible();
            // Only MapPoints within the field of view participate in subsequent projection matching
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        
        // Step 2.2: Perform feature point matching on MapPoints within the field of view through projection
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

/**
 * @brief update LocalMap
 *
 * Local maps include: \n
 * - K1 keyframes, K2 adjacent keyframes and reference keyframes
 * - MapPoints observed by these keyframes
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief update local key points, called by UpdateLocalMap()
 *
 * MapPoints of local key frame mvpLocalKeyFrames, update mvpLocalMapPoints
 */
void Tracking::UpdateLocalPoints()
{
    // Step 1: Clear local MapPoints
    mvpLocalMapPoints.clear();

    // Step 2: Traverse the local keyframes mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        // Step 2: Add the MapPoints of the local keyframes to mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            // mnTrackReferenceForFrame prevents repeated addition of local MapPoint
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


/**
 * @brief update local keyframes, called by UpdateLocalMap()
 *
 * Traverse the MapPoints of the current frame, take out the keyframes and adjacent keyframes where these MapPoints are observed, and update mvpLocalKeyFrames
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // Step 1: Traverse the MapPoints of the current frame and record all the keyframes that can observe the MapPoints of the current frame
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                // Keyframes that can observe the current frame MapPoints
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // Step 2: Update local keyframes (mvpLocalKeyFrames), there are three strategies for adding local keyframes
    // first clear the local keyframe
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        // Strategy 2.1: 10 frames for the best common view
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // Strategy 2.2: own child keyframes
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // Strategy 2.3: own parent keyframe
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    // Step 3: Update the reference key frame of the current frame, and use the key frame with the highest degree of common vision as the reference key frame
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    // Step 1: Calculate the Bow map of the feature points of the current frame
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Step 2: Find candidate keyframes similar to the current frame
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // Step 3: Match by BoW
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // Initialize PnPsolver
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            // Step 4: Estimating pose by EPnP algorithm
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // Step 5: Optimize the pose through PoseOptimization
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // Step 6: If there are fewer interior points, match the previously unmatched points by projection, and then optimize the solution
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
