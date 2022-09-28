#ifndef SYSTEMSETTING_H
#define SYSTEMSETTING_H

#include<string>
#include"ORBVocabulary.h"
#include<opencv2/opencv.hpp>


namespace ORB_SLAM2 {

    class SystemSetting{

        public:
           SystemSetting(ORBVocabulary* voc);

           bool LoadSysSetting(const std::string path);

        public:
           ORBVocabulary* vocavulary;
           float width, height;
           float x, y;
           float wx, hy;
           float b, bf, fps;
           float fx, fy;
           cv::Mat K;
           cv::Mat matDistCoef;
           bool initialized;
           float depth = -1;
           float depthMap = -1;
           float scaleFactor;
           int RGB, features, levels;
           float minThFAST;
           float iniThFAST;
    };
    
}//namespace ORB_SLAM2

#endif //SystemSetting
