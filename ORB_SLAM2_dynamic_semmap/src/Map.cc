/**
* This file is part of ORB-SLAM2 adapted from https://github.com/raulmur/ORB_SLAM2.
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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

******* Modified base code to remove dynamic object and perform semantic mapping for thesis project************
*/

#include "Map.h"

#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

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
    mspCurrentMapPoints.clear();
}

void Map::AddCurrentMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspCurrentMapPoints.insert(pMP);
}

void Map::EraseCurrentMapPoint()
{
    unique_lock<mutex> lock(mMutexMap);
    mspCurrentMapPoints.clear();
}

vector<MapPoint*> Map::GetCurrentMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint *>(mspCurrentMapPoints.begin(), mspCurrentMapPoints.end());
}

void Map::AddObjectMapPoint(int object_id,MapPoint *pMP,KeyFrame* KF)
{
    unique_lock<mutex> lock(mMutexMap);
    cv::Mat Kf_pos = KF->GetPose();
    cv::Mat Rwck(3,3,CV_32F);
    cv::Mat kf_center(3,1,CV_32F);
    Rwck = Kf_pos.rowRange(0,3).colRange(0,3).t();
    kf_center = -Rwck*Kf_pos.rowRange(0,3).col(3);

    cv::Mat pos_i = pMP->GetWorldPos();
    if (abs(sqrt(pow(pos_i.at<float>(0) - kf_center.at<float>(0), 2) + pow(pos_i.at<float>(1) - kf_center.at<float>(1), 2))+ 
                                            pow(pos_i.at<float>(2) - kf_center.at<float>(2), 2)) < 2)
    {
        object_id_points[object_id].push_back(pMP);
    }

    
}

vector<MapPoint*>  Map::GetObjectMapPoint(int object_id)
{
    unique_lock<mutex> lock(mMutexMap);
    //return vector<MapPoint *>(mspCurrentMapPoints.begin(), mspCurrentMapPoints.end());
    return vector<MapPoint*>(object_id_points[object_id].begin(), object_id_points[object_id].end());
}

vector<int> Map::GetObjectid()
{
    unique_lock<mutex> lock(mMutexMap);
    vector<int> object_id_vect;
    for (map<int, vector<MapPoint *>>::const_iterator it = object_id_points.begin(); it!=object_id_points.end();it++)
    {
        object_id_vect.push_back(it->first);
    }
    return object_id_vect;
}

KeyFrame*  Map::GetCommonKeyframe(int object_id)
{
    //unique_lock<mutex> lock(mMutexMap);
    const vector<MapPoint*> &objMP = GetObjectMapPoint(object_id);
    set<MapPoint*> spobjMPs(objMP.begin(), objMP.end());
    vector<int> KFids;
    for(set<MapPoint*>::iterator sit=spobjMPs.begin(), send=spobjMPs.end(); sit!=send; sit++)
    {
        KeyFrame* oKF = (*sit)->GetReferenceKeyFrame();
        KFids.push_back(oKF->mnId);
    }
    return (objMP[findcommonkf(KFids)])->GetReferenceKeyFrame();
}

int Map::findcommonkf(const std::vector<int> &KFids)
{
    vector<int> vect = KFids;
    sort(vect.begin(), vect.end());

    int max = vect[0], result = 0;
    int max_count = 1, res = vect[0], curr_count = 1;
    int n = vect.size();
    for (int i = 1; i < n; i++) {
        if (vect[i] == vect[i - 1])
            curr_count++;
        else {
            if (curr_count > max_count) {
                max_count = curr_count;
                res = vect[i - 1];
            }
            curr_count = 1;
        }
    }
 
    if (curr_count > max_count)
    {
        max_count = curr_count;
        res = vect[n - 1];
    }

    auto it = find(KFids.begin(), KFids.end(), res);
    int index = it - KFids.begin();
    return index;
}

} //namespace ORB_SLAM
