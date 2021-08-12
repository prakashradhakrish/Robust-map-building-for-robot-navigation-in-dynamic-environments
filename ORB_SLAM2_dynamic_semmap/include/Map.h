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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    // Function to manage the current points to color them in green
    void AddCurrentMapPoint(MapPoint *pMP);        // add
    void EraseCurrentMapPoint();                   // delete
    std::vector<MapPoint *> GetCurrentMapPoints(); // get

    void AddObjectMapPoint(int object_id,MapPoint *pMP,KeyFrame* KF);
    std::vector<MapPoint *> GetObjectMapPoint(int object_id);
    KeyFrame* GetCommonKeyframe(int object_id);
    int findcommonkf(const std::vector<int> &KFids);
    std::vector<int> GetObjectid();
    
    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;
    std::set<MapPoint *> mspCurrentMapPoints;
    std::map<int,std::vector<MapPoint*>> object_id_points;
    


    std::vector<MapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
