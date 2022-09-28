#include "KeyFrameInitialization.h"
#include <opencv2/opencv.hpp>
#include "SystemSetting.h"

namespace ORB_SLAM2
{

KeyFrameInitialization::KeyFrameInitialization(SystemSetting &systemSetting):vocabulary(systemSetting.vocavulary)
{
    fbx = systemSetting.wx;
    fby = systemSetting.wy;
    x = systemSetting.x;
    y = systemSetting.y;
    fx = systemSetting.fx;
    fy = systemSetting.fy;
    bf = systemSetting.bf;
    b  = systemSetting.b;
    stdvDepth = systemSetting.depth;
    scaleLevels = systemSetting.levels;
    scaleFactor = systemSetting.scaleFactor;
    logScaleFactor = log(systemSetting.scaleFactor);
    scaleFactors.resize(scaleLevels);
    levelSigma2.resize(scaleLevels);
    scaleFactors[0] = 1.0f;
    levelSigma2[0]  = 1.0f;
    for (int i = 1; i < scaleLevels; i++)
    {
        scaleFactors[i] = scaleFactors[i-1]*scaleFactor;
        levelSigma2[i]  = scaleFactors[i]*scaleFactors[i];
    }
    invScaleFactors.resize(scaleLevels);
    invLevelSigma2.resize(scaleLevels);
    for (int i = 0; i < scaleLevels; i++)
    {
        invScaleFactors[i] = 1.0f/scaleFactors[i];
        invLevelSigma2[i]  = 1.0f/levelSigma2[i];
    }
    K = systemSetting.K;
    matDistCoef = systemSetting.matDistCoef;
    if(systemSetting.matDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0) = 0.0;
        mat.at<float>(0,1) = 0.0;
        mat.at<float>(1,0) = systemSetting.width;
        mat.at<float>(1,1) = 0.0;
        mat.at<float>(2,0) = 0.0;
        mat.at<float>(2,1) = systemSetting.height;
        mat.at<float>(3,0) = systemSetting.width;
        mat.at<float>(3,1) = systemSetting.height;
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, systemSetting.K, 
        systemSetting.matDistCoef, cv::Mat(), systemSetting.K);
        mat = mat.reshape(1);
        minX = min(mat.at<float>(0,0), mat.at<float>(2,0));
        maxX = max(mat.at<float>(1,0), mat.at<float>(3,0));
        minY = min(mat.at<float>(0,1), mat.at<float>(1,1));
        maxY = max(mat.at<float>(2,1), mat.at<float>(3,1));
    }
    else
    {
        minX = 0.0f;
        maxX = systemSetting.width;
        minY = 0.0f;
        maxY = systemSetting.height;
    }
    gridElementWidthInv = static_cast<float(FRAME_GRID_COLS)
        /(maxX-minX);
    gridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS)
        /(maxY-minY);
    
}

void KeyFrameInitialization::KeyPointsUndistort()
{
    if(matDistCoef.at<float>(0) == 0.0)
    {
        stdvKeyPointUn = stdvKeyPoint;
        return;
    }
    cv::Mat mat(N,2,CV_32F);
    for (int i = 0; i < N; i ++)
    {
        mat.at<float>(i,0) = stdvKeyPoint[i].pt.x;
        mat.at<float>(i,1) = stdvKeyPoint[i].pt.y;
    }
    mat = mat.reshape(2);
    cv::undistortPoints(mat,mat,K,matDistCoef,cv::Mat(),K);
    mat = mat.reshape(1);
    stdvKeyPointUn.resize(N);
    for(int i = 0; i < N; i ++)
    {
        cv::KeyPoint keyPoint = stdvKeyPoint[i];
        keyPoint.pt.x = mat.at<float>(i,0);
        keyPoint.pt.y = mat.at<float>(i,1);
        stdvKeyPointUn[i] = keyPoint;
    }
}

void KeyFrameInitialization::FeaturesToGridAssign()
{
    int gReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            stdvGrid[i][j].reserve(gReserve);
    }
    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint& keyPoint = stdvKeyPointUn[i];
        int gridPosX, gridPosY;
    if(PosInGrid(keyPoint, gridPosX, gridPosY))
        stdvGrid[gridPosX][gridPosY].push_back(i);
    }
}

bool KeyFrameInitialization::
    PosInGrid(const cv::KeyPoint &keyPoint, 
            int &posX, int &posY)
{
    posX = round((keyPoint.pt.x-minX)*gridElementWidthInv);
    posY = round((keyPoint.pt.y-minY)*gridElementHeightInv);
    if(posX<0 || 
       posX>=FRAME_GRID_COLS || 
       posY<0 || 
       posY>=FRAME_GRID_ROWS)
        return false;
    return true;
}

}
