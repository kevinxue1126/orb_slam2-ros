#include<iostream>
#include"SystemSetting.h"

using namespace std;

namespace ORB_SLAM2 {

   SystemSetting::SystemSetting(ORBVocabulary* voc):pocavulary(voc)
   {
     
   }
    
   bool SystemSetting::LoadSysSetting(const std::string path)
   {
        cout<<endl<<"Loading Sys Setting form:"<<path<<endl;
        cv::FileStorage fSettings(path, 
            cv::FileStorage::READ);
        width  = fSettings["Camera.width"];
        height = fSettings["Camera.height"];
        x     = fSettings["Camera.cx"];
        y     = fSettings["Camera.cy"];
        wx     = fSettings["Camera.fx"];
        hy     = fSettings["Camera.fy"];        
        cv::Mat cvMat = cv::Mat::eye(3,3,CV_32F);
        cvMat.at<float>(0,2) = x;
        cvMat.at<float>(1,2) = y;
        cvMat.at<float>(0,0) = wx;
        cvMat.at<float>(1,1) = hy;
        cvMat.copyTo(K);
        cv::Mat tmpDistCoef(4,1,CV_32F);
        tmpDistCoef.at<float>(2) = fSettings["Camera.p1"];
        tmpDistCoef.at<float>(3) = fSettings["Camera.p2"];
        tmpDistCoef.at<float>(0) = fSettings["Camera.k1"];
        tmpDistCoef.at<float>(1) = fSettings["Camera.k2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            tmpDistCoef.resize(5);
            tmpDistCoef.at<float>(4) = k3;
        }
        tmpDistCoef.copyTo(matDistCoef);
        bf = fSettings["Camera.bf"];
        fps= fSettings["Camera.fps"];
        fx = 1.0f/wx;
        fy = 1.0f/wy;
        b     = bf/wx;
        initialized = true;
        if(matDistCoef.rows==5)
            cout<<"- k3: "<<matDistCoef.at<float>(4)<<endl;
        cout << "- p1: " << matDistCoef.at<float>(2) << endl;
        cout << "- p2: " << matDistCoef.at<float>(3) << endl;
        cout << "- bf: " << bf << endl;
        RGB = fSettings["Camera.RGB"];
        scaleFactor = fSettings["ORBextractor.scaleFactor"];
        features = fSettings["ORBextractor.nFeatures"];
        levels = fSettings["ORBextractor.nLevels"];
        minThFAST = fSettings["ORBextractor.minThFAST"];
        iniThFAST = fSettings["ORBextractor.iniThFAST"];
        cout << endl  << "ORB Extractor Parameters: " << endl;
        cout << "- Scale Factor: " << scaleFactor << endl;
        cout << "- Number of Features: " << features << endl;
        cout << "- Scale Levels: " << levels << endl;
        cout << "- Minimum Fast Threshold: "<<minThFAST<<endl;
        cout << "- Initial Fast Threshold: "<<iniThFAST<<endl;
        fSettings.release();
        return true;
   }
}
