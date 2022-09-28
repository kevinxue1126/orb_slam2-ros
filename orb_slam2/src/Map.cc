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

#include "Map.h"

#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}



KeyFrame* Map::LoadKeyFrame(ifstream &fin, SystemSetting* systemSetting)
{
    KeyFrameInitialization kfInit(*systemSetting);
    fin.read((char*)&kfInit.nId, sizeof(kfInit.nId));
    fin.read((char*)&kfInit.timeStampNum, sizeof(double));
    cv::Mat cvMat = cv::Mat::zeros(4,4,CV_32F);
    std::vector<float> stdQuat(4);
    
    for (int i = 0; i < 4; i++)
        fin.read((char*)&stdQuat[i],sizeof(float));
    cv::Mat cvMatChild = Converter::toCvMat(stdQuat);
    for (int i = 0; i < 3; i++)
        fin.read((char*)&T.at<float>(i,3),sizeof(float));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cvMat.at<float>(i,j) = cvMatChild.at<float>(i,j);
    
    cvMat.at<float>(3,3) = 1;
    fin.read((char*)&kfInit.N, sizeof(kfInit.N));
    kfInit.vKps.reserve(kfInit.N);
    kfInit.descriptors.create(kfInit.N, 32, CV_8UC1);
    vector<float>KeypointDepth;
    std::vector<MapPoint*> vpMapPoints;
    vpMapPoints = vector<MapPoint*>(kfInit.N, static_cast<MapPoint*>(NULL));
    
    std::vector<MapPoint*> vAllMapPoints = GetAllMapPoints();
    
    for(int i = 0; i < kfInit.N; i++)
    {
        cv::KeyPoint keyPoint;
        fin.read((char*)&keyPoint.pt.x, sizeof(keyPoint.pt.x));
        fin.read((char*)&keyPoint.pt.y, sizeof(keyPoint.pt.y));
        fin.read((char*)&keyPoint.size, sizeof(keyPoint.size));
        fin.read((char*)&keyPoint.angle, sizeof(keyPoint.angle));
        fin.read((char*)&keyPoint.response, sizeof(keyPoint.response));
        fin.read((char*)&keyPoint.octave, sizeof(keyPoint.octave));
        kfInit.vKps.push_back(keyPoint);
        fin.read((char*)&kfInit.descriptors.cols, sizeof(kfInit.descriptors.cols));
        
        for (int j = 0; j < kfInit.descriptors.cols; j++)
            fin.read((char*)&kfInit.descriptors.at<unsigned char>(i,j),sizeof(char));
        
        unsigned long int mpIndex;
        f.read((char*)&mpIndex, sizeof(mpidx));
        if(mpIndex == ULONG_MAX)
            vpMapPoints[i] = NULL;
        else
            vpMapPoints[i] = vAllMapPoints[mpIndex];
    }
    
    kfInit.vRight = vector<float>(kfInit.N,-1);
    kfInit.vDepth = vector<float>(kfInit.N,-1);
    kfInit.UndistortKeyPoints();
    kfInit.AssignFeaturesToGrid();
    KeyFrame* keyFrame = new KeyFrame(kfInit, this, NULL, vpMapPoints);
    keyFrame->mnId = kfInit.nId;
    keyFrame->SetPose(cvMat);
    keyFrame->ComputeBoW();
    
    for (int i = 0; i < kfInit.N; i++)
    {
        if (vpMapPoints[i])
        {
            vpMapPoints[i]->AddObservation(keyFrame,i);
            if(!vpMapPoints[i]->GetReferenceKeyFrame())
                vpMapPoints[i]->SetReferenceKeyFrame(keyFrame);
        }
    }
    
    return keyFrame;
}



MapPoint* Map::LoadMapPoint(ifstream &fin)
{
    cv::Mat Position(3,1,CV_32F);
    long unsigned int id;
    fin.read((char*)&id, sizeof(id));
    fin.read((char*)&Position.at<float>(0), sizeof(float));
    fin.read((char*)&Position.at<float>(1), sizeof(float));
    fin.read((char*)&Position.at<float>(2), sizeof(float));
    
    MapPoint* mapPoint = new MapPoint(Position, this);
    mapPoint->mnId = id;
    mapPoint->SetWorldPos(Position);
    
    return mapPoint;
}



void Map::LoadMap(const string &fileName, SystemSetting* systemSetting)
{
    cerr << "Map.cc :: Map load from:"<<fileName<<endl;
    
    ifstream fin;
    fin.open(fileName.c_str(),ios::binary);
    unsigned long int mapPoints;
    fin.read((char*)&mapPoints, sizeof(mapPoints));
    
    cerr<<"Map.cc :: mapPoints is :"<<mapPoints<<endl;
    
    for (unsigned int i = 0; i < mapPoints; i++)
    {
        MapPoint* mapPoint = LoadMapPoint(f);
        AddMapPoint(mapPoint);
    }
    
    unsigned long int keyFrames;
    fin.read((char*)&keyFrames, sizeof(keyFrames));
    cerr<<"Map.cc :: keyFrames is :"<<keyFrames<<endl;
    vector<KeyFrame*>kfByOrder;
    for(unsigned int i = 0; i < keyFrames; i++)
    {
        KeyFrame* keyFrame = LoadKeyFrame(fin, systemSetting);
        AddKeyFrame(keyFrame);
        kfByOrder.push_back(keyFrame); 
    }
    
    cerr<<"Map.cc :: KeyFrame Load Finished!"<<endl;
    
    map<unsigned long int, KeyFrame*> kfById;
    for (auto keyFrame: mspKeyFrames)
        kfById[keyFrame->mnId] = keyFrame;
    cerr<<"Map.cc :: Map Start Load The Parent!"<<endl;
    for(auto kf: kfByOrder)
    {
        unsigned long int parentId;
        fin.read((char*)&parentId, sizeof(parentId));
        if (parentId != ULONG_MAX)
            kf->ChangeParent(kfById[parentId]);
        
        unsigned long int con;
        fin.read((char*)&con, sizeof(con));
        for (unsigned long int i = 0; i < con; i++)
        {
            unsigned long int id;
            int weight;
            fin.read((char*)&id, sizeof(id));
            fin.read((char*)&weight, sizeof(weight));
            kf->AddConnection(kfById[id],weight);
        }
   }
    
   cerr<<"Map.cc :: Map Parent Load Finished!"<<endl;
   std::vector<MapPoint*> vAllMapPoints = GetAllMapPoints();
   for (auto mapPoint: vAllMapPoints)
   {
       if(mapPoint)
       {
            mapPoint->ComputeDistinctiveDescriptors();
            mapPoint->UpdateNormalAndDepth();
        }
   }
    fin.close();
    cerr<<"Map.cc :: Load Map Finished!"<<endl;
    return;
}



void Map::SaveMap(const string& fileName)
{
    cerr<<"Map.cc :: SaveMap to "<<fileName <<endl;
    
    ofstream fout;
    fout.open(fileName.c_str(), ios_base::out|ios::binary);
    unsigned long int mapPointsSize = mspMapPoints.size();
    fout.write((char*)&mapPointsSize, sizeof(mapPointsSize));
    
    for (auto mapPoint: mspMapPoints){
        SaveMapPoint(fout, mapPoint);
    }
    
    cerr << "Map.cc::mapPointsSize is :"<<mapPointsSize<<endl;
    
    GetMapPointsIndex(); 
    
    unsigned long int keyFramesSize = mspKeyFrames.size();
    cerr <<"Map.cc::keyFramesSize is :"<<keyFramesSize<<endl;
    fout.write((char*)&keyFramesSize, sizeof(keyFramesSize));
    
    for (auto keyFrame: mspKeyFrames)
        SaveKeyFrame(fout, keyFrame);
    
    for (auto keyFrame:mspKeyFrames)
    {
        KeyFrame* kfParent = keyFrame->GetParent();
        unsigned long int parentId = ULONG_MAX;
        if (kfParent)
            parentId = kfParent->mnId;
        fout.write((char*)&parentId, sizeof(parentId));
        unsigned long int ckfSize = keyFrame->GetConnectedKeyFrames().size();
        fout.write((char*)&ckfSize, sizeof(ckfSize));
        for (auto ckf: keyFrame->GetConnectedKeyFrames())
        {
            int weight = keyFrame->GetWeight(ckf);
            fout.write((char*)&ckf->mnId, sizeof(ckf->mnId));
            fout.write((char*)&weight, sizeof(weight));
        }
    }
    
    fout.close();
    
    cerr<<"Map.cc :: Map Saving Map Finished!"<<endl;
}


void Map::SaveMapPoint(ofstream& fout, MapPoint* mapPoint)
{   
    fout.write((char*)&mapPoint->mnId, sizeof(mapPoint->mnId));
    cv::Mat matWorldPos = mapPoint->GetWorldPos();
    fout.write((char*)& matWorldPos.at<float>(0), sizeof(float));
    fout.write((char*)& matWorldPos.at<float>(1), sizeof(float));
    fout.write((char*)& matWorldPos.at<float>(2), sizeof(float));
}
    


void Map::SaveKeyFrame(ofstream &fout, KeyFrame* keyFrame)
{
    
    fout.write((char*)&keyFrame->mnId, sizeof(keyFrame->mnId));
    fout.write((char*)&keyFrame->mTimeStamp, sizeof(keyFrame->mTimeStamp));
    
    cv::Mat matPose = keyFrame->GetPose();    
    std::vector<float> stdQ = Converter::toQuaternion(matPose);
    for (int i = 0; i < 3; i ++)
        fout.write((char*)&matPose.at<float>(i,3), sizeof(float));
    for (int i = 0; i < 4; i ++)
        fout.write((char*)&stdQ[i],sizeof(float));
    
    fout.write((char*)&keyFrame->N, sizeof(keyFrame->N));
    for(int i = 0; i < keyFrame->N; i++)
    {
        cv::KeyPoint keyPoint = keyFrame->mvKeys[i];
        fout.write((char*)&keyPoint.pt.x, sizeof(keyPoint.pt.x));
        fout.write((char*)&keyPoint.pt.y, sizeof(keyPoint.pt.y));
        fout.write((char*)&keyPoint.size, sizeof(keyPoint.size));
        fout.write((char*)&keyPoint.angle, sizeof(keyPoint.angle));
        fout.write((char*)&keyPoint.response, sizeof(keyPoint.response));
        fout.write((char*)&keyPoint.octave, sizeof(keyPoint.octave));
        fout.write((char*)&keyPoint->mDescriptors.cols, 
        sizeof(keyFrame->mDescriptors.cols)); 
        
        for (int j = 0; j < keyFrame->mDescriptors.cols; j++)
            fout.write((char*)&keyFrame->mDescriptors.at<unsigned char>(i,j), sizeof(char));
        unsigned long int mnIndex;
        MapPoint* mapPoint = keyFrame->GetMapPoint(i);
        if (mapPoint == NULL)
            mnIndex = ULONG_MAX;
        else
            mnIndex = mapPointsIndex[mapPoint];
        
        fout.write((char*)&mnIndex, sizeof(mnIndex));
    }
}



void Map::GetMapPointsIndex()
{
    unique_lock<mutex> lock(mMutexMap);
    unsigned long int i = 0;
    for (auto mapPoint: mspMapPoints)
    {
        mapPointsIndex[mapPoint] = i;
        i += 1;
    }
}

    

/**
 * @brief Insert KeyFrame in the map
 * @param pKF KeyFrame
 */
void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

/**
 * @brief Insert MapPoint in the map
 * @param pMP MapPoint
 */
void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

/**
 * @brief Erase MapPoint from the map
 * @param pMP MapPoint
 */
void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

/**
 * @brief Erase KeyFrame from the map
 * @param pKF KeyFrame
 */
void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

    
/**
 * @brief Set the reference MapPoints, which will be used for drawing by the DrawMapPoints function
 * @param vpMPs Local MapPoints
 */
void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

// map serialization addition
template<class Archive>
void Map::serialize(Archive &ar, const unsigned int version)
{
    // don't save mutex
    unique_lock<mutex> lock_MapUpdate(mMutexMapUpdate);
    unique_lock<mutex> lock_Map(mMutexMap);
    ar & mspMapPoints;
    ar & mvpKeyFrameOrigins;
    ar & mspKeyFrames;
    ar & mvpReferenceMapPoints;
    ar & mnMaxKFid & mnBigChangeIdx;
}
template void Map::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void Map::serialize(boost::archive::binary_oarchive&, const unsigned int);

} //namespace ORB_SLAM
