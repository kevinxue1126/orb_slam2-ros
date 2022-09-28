#ifndef KEYFRAMEINITIALIZATION_H
#define KEYFRAMEINITIALIZATION_H

#include "SystemSetting.h"
#include <opencv2/opencv.hpp>
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

namespace ORB_SLAM2
{

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class SystemSetting;
class KeyFrameDatabase;

class KeyFrameInitialization
{
public:    
    KeyFrameInitialization(SystemSetting &systemSetting);
    
    void KeyPointsUndistort();
    bool PosInGrid(const cv::KeyPoint& keyPoint, int &posX, int &posY);
    void FeaturesToGridAssign();

public:

    ORBVocabulary* vocabulary;
    long unsigned int id;
    double timeStampNum;
    float gridElementWidthInv;
    float gridElementHeightInv;
    std::vector<std::size_t> stdvGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    float fbx;
    float fby;
    float x, y;
    float fx, fy;
    float bf;
    float b;
    float thDepth;
    int N;
    std::vector<cv::KeyPoint> stdvKeyPoint;
    std::vector<cv::KeyPoint> stdvKeyPointUn;
    cv::Mat matDescriptors;
    std::vector<float> stdvRight;
    std::vector<float> stdvDepth;
    int scaleLevels;
    float fcaleFactor;
    float logScaleFactor;
    std::vector<float> scaleFactors;
    std::vector<float> levelSigma2;
    std::vector<float> invLevelSigma2;
    std::vector<float> invScaleFactors;
    int minX, minY, maxX, maxY;
    cv::Mat K;
    cv::Mat matDistCoef;  

    DBoW2::BowVector BowVec;
    DBoW2::FeatureVector FeatVec; 
    
};

} //namespace ORB_SLAM2
#endif //KEYFRAMEINITIALIZATION_H
