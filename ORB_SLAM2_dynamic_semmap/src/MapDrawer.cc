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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <cmath>

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

}


void MapDrawer::DrawMapPoints(const bool bDrawCurrentPoints, 
                                const bool bDrawdynamicinfo, 
                                const bool staticpoints_ON, 
                                const bool non_mvng_dyn_ON, 
                                const bool mvng_dyn_ON,
                                pangolin::OpenGlMatrix &Twc,
                                const bool bcuboid)
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();
    const vector<MapPoint*> &vpCurrentMPs = mpMap->GetCurrentMapPoints(); 
    // Added for object id mapping
    const vector<int> obj_id_list = mpMap->GetObjectid();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());
    set<MapPoint*> spCurrMPs(vpCurrentMPs.begin(), vpCurrentMPs.end());
    bool dynamic_mapping = false;
    int dynamic_map_points = 0;

    float min_x=0, min_z=0, max_x=0, max_z=0, center_x=0, center_y = 0, center_z=0;
    if (!mCameraPose.empty())
    {
        cv::Mat Rwce(3,3,CV_32F);
        cv::Mat current_center(3,1,CV_32F);
        Rwce = mCameraPose.rowRange(0,3).colRange(0,3).t();
        current_center = -Rwce*mCameraPose.rowRange(0,3).col(3);

        min_x = current_center.at<float>(2) + 1;
        min_z = current_center.at<float>(2) - 1;
        max_x = current_center.at<float>(0) + 1;
        max_z = current_center.at<float>(2) + 1;
        center_x = current_center.at<float>(0);
        center_y = current_center.at<float>(1);
        center_z = current_center.at<float>(2);
    }

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);

    vector<vector<MapPoint*>> obj_map_ref;
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            {//std::cout<<"bad 1"<<std::endl;
            continue;}
        //std::cout<<"good 1"<<std::endl;
        cv::Mat pos = vpMPs[i]->GetWorldPos();

       if (dynamic_mapping) // Changes the color of static map based on dynamic count
       {
          if( (pos.at<float>(0) < max_x) && (pos.at<float>(0)>min_x) && (pos.at<float>(2)<max_z) && (pos.at<float>(2)>min_z) && (!mCameraPose.empty()))
          {
             KeyFrame* kf_curr = vpMPs[i]->GetReferenceKeyFrame();
             if (kf_curr->dynamic_counter < 2)
             { 
             glColor3f(0.0,1.0,0.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
             }
             if (2 <= kf_curr->dynamic_counter && kf_curr->dynamic_counter <= 5)
             { 
             glColor3f(1.0,0.5,0.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
             }
             if (5 < kf_curr->dynamic_counter)
             { 
             glColor3f(1.0,0.0,0.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
             }   
          }
          else
          {
             glColor3f(1.0,1.0,1.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
          }
        }
        else
        {
            if(staticpoints_ON)
            {
                if(vpMPs[i]->dynamic_id==0)
                {
                    glColor3f(0.0,1.0,0.0);
                    if(pos.at<float>(1)<0.1)
                    {
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    
                }
            }
            if(non_mvng_dyn_ON)
            {
                //int obj_ref_id = vpMPs[i]->object_id;
                if(vpMPs[i]->dynamic_id==1 && vpMPs[i]->object_id>0)
                {
                    //adding points to the vector for centroid
                    /*if (obj_map_ref.size() >= obj_ref_id)
                    {
                        obj_map_ref[obj_ref_id-1].push_back(vpMPs[i]);
                    } 
                    else
                    {
                        int diff  = obj_ref_id - obj_map_ref.size();
                        for(int i=0; i<diff; i++)
                        {
                            obj_map_ref.push_back(vector<MapPoint*>(1,static_cast<MapPoint*>(NULL)));
                        }
                        obj_map_ref[obj_ref_id-1].push_back(vpMPs[i]);
                    }*/

                    int color_gen = vpMPs[i]->object_id % 10;
                    //std::cout<<"Color generation"<< color_gen<<std::endl;
                    if (color_gen == 0)
                    {
                        //glColor3f(1.0,0.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 1)
                    {
                        //glColor3f(0.0,1.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 2)
                    {
                        //glColor3f(0.0,0.0,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 3)
                    {
                        //glColor3f(1.0,1.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 4)
                    {
                        //glColor3f(1.0,0.0,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    } 
                    if (color_gen == 5)
                    {
                        //glColor3f(0.0,1.0,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 6)
                    {
                        //glColor3f(1.0,0.5,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 7)
                    {
                        //glColor3f(0.5,1.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 8)
                    {
                        //glColor3f(0.5,1.0,0.5);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 9)
                    {
                        //glColor3f(0.0,0.5,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }      
                    
                }
            }
            if(mvng_dyn_ON)
            {
                if(vpMPs[i]->dynamic_id==2 && vpMPs[i]->object_id>0)
                {
                   // int obj_ref_id = vpMPs[i]->object_id;
                    //adding points to the vector for centroid
                    /*if (obj_map_ref.size() >= obj_ref_id)
                    {
                        obj_map_ref[obj_ref_id-1].push_back(vpMPs[i]);
                    } 
                    else
                    {
                        int diff  = obj_ref_id - obj_map_ref.size();
                        for(int i=0; i<diff; i++)
                        {
                            obj_map_ref.push_back(vector<MapPoint*>(1,static_cast<MapPoint*>(NULL)));
                        }
                        obj_map_ref[obj_ref_id-1].push_back(vpMPs[i]);
                    }*/

                    int color_gen = vpMPs[i]->object_id % 10;
                    //std::cout<<"Color generation"<< color_gen<<std::endl;
                    if (color_gen == 0)
                    {
                        //glColor3f(1.0,0.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 1)
                    {
                        //glColor3f(0.0,1.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 2)
                    {
                        //glColor3f(0.0,0.0,1.0);
                        glColor3f(0.0,0.0,1.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 3)
                    {
                        //glColor3f(1.0,1.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 4)
                    {
                        //glColor3f(1.0,0.0,1.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    } 
                    if (color_gen == 5)
                    {
                        //glColor3f(0.0,1.0,1.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 6)
                    {
                        //glColor3f(1.0,0.5,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 7)
                    {
                        //glColor3f(0.5,1.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 8)
                    {
                        //glColor3f(0.5,1.0,0.5);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 9)
                    {
                        //glColor3f(0.0,0.5,1.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }    
                if ((pow(pos.at<float>(0)-center_x,2) + pow(pos.at<float>(1)-center_y,2) + pow(pos.at<float>(2)-center_z,2)) <= pow(0.5,2))
                {
                    dynamic_map_points++;
                }
                }
            }
        }
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        
        if((*sit)->isBad())
            {//std::cout<<"bad 2"<<std::endl;
            continue;}
        //std::cout<<"good 2"<<std::endl;
        cv::Mat pos = (*sit)->GetWorldPos();
       if (dynamic_mapping)
       {
          if( (pos.at<float>(0) < max_x) && (pos.at<float>(0)>min_x) && (pos.at<float>(2)<max_z) && (pos.at<float>(2)>min_z) && (!mCameraPose.empty()))
          {
          KeyFrame* kf_curr = (*sit)->GetReferenceKeyFrame();
          if (kf_curr->dynamic_counter < 2)
          { 
             glColor3f(0.0,1.0,0.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
          }
          if (2 <= kf_curr->dynamic_counter && kf_curr->dynamic_counter <= 5)
          { 
             glColor3f(1.0,0.5,0.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
          }
          if (5 < kf_curr->dynamic_counter)
          { 
             glColor3f(1.0,0.0,0.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
          }    
          }
          else
          {
             glColor3f(1.0,1.0,1.0);
             glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
          }
       }
       else
        {
            if(staticpoints_ON)
            {
                if((*sit)->dynamic_id==0)
                {
                    glColor3f(0.0,1.0,0.0);
                    if(pos.at<float>(1)<0.1)
                    {
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                }
            }
            if(non_mvng_dyn_ON)
            {
                if((*sit)->dynamic_id==1 && (*sit)->object_id>0)
                {
                    //if ((*sit)->object_id<20)   
                    {  
                    int color_gen = (*sit)->object_id % 10;
                    //std::cout<<"Color generation"<< color_gen<<std::endl;
                    if (color_gen == 0)
                    {
                        //glColor3f(1.0,0.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 1)
                    {
                        //glColor3f(0.0,1.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 2)
                    {
                        //glColor3f(0.0,0.0,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 3)
                    {
                        //glColor3f(1.0,1.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 4)
                    {
                        //glColor3f(1.0,0.0,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    } 
                    if (color_gen == 5)
                    {
                        //glColor3f(0.0,1.0,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 6)
                    {
                        //glColor3f(1.0,0.5,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 7)
                    {
                        //glColor3f(0.5,1.0,0.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 8)
                    {
                        //glColor3f(0.5,1.0,0.5);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 9)
                    {
                        //glColor3f(0.0,0.5,1.0);
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }   
                    }
                    
                }
            }
            if(mvng_dyn_ON)
            {
                if((*sit)->dynamic_id==2)
                {
                    //if ((*sit)->object_id<20)   
                    {  
                    //std::cout<<"hello1"<<std::endl;
                    int color_gen = (*sit)->object_id % 10;
                    //std::cout<<"Color generation"<< color_gen<<std::endl;
                    if (color_gen == 0)
                    {
                        //glColor3f(1.0,0.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 1)
                    {
                        //glColor3f(0.0,1.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 2)
                    {
                        //glColor3f(0.0,0.0,1.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 3)
                    {
                        //glColor3f(1.0,1.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }
                    if (color_gen == 4)
                    {
                        //glColor3f(1.0,0.0,1.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    } 
                    if (color_gen == 5)
                    {
                        //glColor3f(0.0,1.0,1.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 6)
                    {
                        //glColor3f(1.0,0.5,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 7)
                    {
                        //glColor3f(0.5,1.0,0.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 8)
                    {
                        //glColor3f(0.5,1.0,0.5);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    }  
                    if (color_gen == 9)
                    {
                        //glColor3f(0.0,0.5,1.0);
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                    } 
                    }
                if ((pow(pos.at<float>(0)-center_x,2) + pow(pos.at<float>(1)-center_y,2) + pow(pos.at<float>(2)-center_z,2)) <= pow(0.5,2))
                {
                    dynamic_map_points++;
                }
                }
            }
      
        }

    }
    glEnd();

    if(bcuboid)
    {
        glPointSize(10);
        glBegin(GL_POINTS);                 
        for(int i=1;i < obj_id_list.size();i++)
        //for(int i=0; i<obj_map_ref.size();i++ )
        {
            //std::cout<<"Map drawer"<<std::endl;
            float x=0;float y=0;float z=0;
            const vector<MapPoint*> &objMPs = mpMap->GetObjectMapPoint(obj_id_list[i]);
            //const vector<MapPoint*> &objMPs = obj_map_ref[i];
            set<MapPoint*> spobjMPs(objMPs.begin(), objMPs.end());

            /*KF based constraint 
            KeyFrame* kf_objref = mpMap->GetCommonKeyframe(obj_id_list[i]);
            cv::Mat Kf_pos = kf_objref->GetPose();
            cv::Mat Rwck(3,3,CV_32F);
            cv::Mat kf_center(3,1,CV_32F);
            Rwck = Kf_pos.rowRange(0,3).colRange(0,3).t();
            kf_center = -Rwck*Kf_pos.rowRange(0,3).col(3);

            float kf_x = kf_center.at<float>(0);
            float kf_y = kf_center.at<float>(1);
            float kf_z = kf_center.at<float>(2); */
            

            int counter = 0 ;
            float min_x=9999,min_y=9999,min_z=9999,max_x=-9999,max_y=-9999,max_z=-9999;
            if (spobjMPs.size()>10)
            {
                for(set<MapPoint*>::iterator sit=spobjMPs.begin(), send=spobjMPs.end(); sit!=send; sit++)
                {

                    cv::Mat pos_i = (*sit)->GetWorldPos();
                    /*float dist_bw_pts = 9999;
                    float dist_thresh = 1;
                    for(set<MapPoint*>::iterator sjt=spobjMPs.begin(), send=spobjMPs.end(); sjt!=send; sjt++)
                    {
                        cv::Mat pos_j = (*sjt)->GetWorldPos();
                        dist_bw_pts = (sqrt(pow(pos_i.at<float>(0) - pos_j.at<float>(0), 2) + pow(pos_i.at<float>(1) - pos_j.at<float>(1), 2))+ 
                                            pow(pos_i.at<float>(2) - pos_j.at<float>(2), 2));
                        if (dist_bw_pts<=dist_thresh)
                            break;
                    }
                    //std::cout<<"map ref"<<std::endl;
                    if ((*sit)->isBad())
                        continue;*/
                                   
                    //if ((sqrt(pow(x - kf_x, 2) + pow(y - kf_y, 2))+ pow(z - kf_z, 2)) < 50)
                    //if (dist_bw_pts<=dist_thresh)
                    {
                        x += pos_i.at<float>(0);
                        y += pos_i.at<float>(1);
                        z += pos_i.at<float>(2);

                        //Finding extreme points for drawing cuboid
                        if(pos_i.at<float>(0)<min_x)
                        {
                            min_x = pos_i.at<float>(0);
                        }
                        if(pos_i.at<float>(1)<min_y)
                        {
                            min_y = pos_i.at<float>(1);
                        }
                        if(pos_i.at<float>(2)<min_z)
                        {
                            min_z = pos_i.at<float>(2);
                        }
                        if(pos_i.at<float>(0)>max_x)
                        {
                            max_x = pos_i.at<float>(0);
                        }
                        if(pos_i.at<float>(1)>max_y)
                        {
                            max_y = pos_i.at<float>(1);
                        }
                        if(pos_i.at<float>(2)>max_z)
                        {
                            max_z = pos_i.at<float>(2);
                        }
                        counter ++;
                    }               
                }

                if(counter>20)
                {
                    x = x/counter;
                    y = y/counter;
                    z = z/counter;
                    int color_gen = i % 10;
                    //std::cout<<"Color generation"<< color_gen<<std::endl;
                    if (color_gen == 0)
                    {
                        glColor3f(1.0,0.0,0.0);
                        glVertex3f(x,y,z);
                    }
                    if (color_gen == 1)
                    {
                        glColor3f(0.0,1.0,0.0);
                        glVertex3f(x,y,z);
                    }
                    if (color_gen == 2)
                    {
                        glColor3f(0.0,0.0,1.0);
                        glVertex3f(x,y,z);
                    }
                    if (color_gen == 3)
                    {
                        glColor3f(1.0,1.0,0.0);
                        glVertex3f(x,y,z);
                    }
                    if (color_gen == 4)
                    {
                        glColor3f(1.0,0.0,1.0);
                        glVertex3f(x,y,z);
                    } 
                    if (color_gen == 5)
                    {
                        glColor3f(0.0,1.0,1.0);
                        glVertex3f(x,y,z);
                    }  
                    if (color_gen == 6)
                    {
                        glColor3f(1.0,0.5,0.0);
                        glVertex3f(x,y,z);
                    }  
                    if (color_gen == 7)
                    {
                        glColor3f(0.5,1.0,0.0);
                        glVertex3f(x,y,z);
                    }  
                    if (color_gen == 8)
                    {
                        glColor3f(0.5,1.0,0.5);
                        glVertex3f(x,y,z);
                    }  
                    if (color_gen == 9)
                    {
                        glColor3f(0.0,0.5,1.0);
                        glVertex3f(x,y,z);
                    } 
                    //glVertex3f(x,y,z);
                    //glPushMatrix();
                    //glTranslatef(x,y,z);
                    //pangolin::glDrawColouredCube(-0.05,0.05);
                    //glPopMatrix();
                    //Draw cuboid axis aligned
                    /*glColor3f(1.0, 0.0, 0.0);
                    glBegin(GL_LINES);
                    // Bottom half
                    glVertex3d(min_x,min_y,min_z);
                    glVertex3d(max_x,min_y,min_z);

                    glVertex3d(max_x,min_y,min_z);
                    glVertex3d(max_x,min_y,max_z);

                    glVertex3d(max_x,min_y,max_z);
                    glVertex3d(min_x,min_y,max_z);

                    glVertex3d(min_x,min_y,max_z);
                    glVertex3d(min_x,min_y,min_z);

                    // Top half

                    glVertex3d(min_x,max_y,min_z);
                    glVertex3d(max_x,max_y,min_z);

                    glVertex3d(max_x,max_y,min_z);
                    glVertex3d(max_x,max_y,max_z);

                    glVertex3d(max_x,max_y,max_z);
                    glVertex3d(min_x,max_y,max_z);

                    glVertex3d(min_x,max_y,max_z);
                    glVertex3d(min_x,max_y,min_z);

                    // Connection between bottom and top

                    glVertex3d(min_x,min_y,min_z);
                    glVertex3d(min_x,max_y,min_z);

                    glVertex3d(max_x,min_y,min_z);
                    glVertex3d(max_x,max_y,min_z);

                    glVertex3d(max_x,min_y,max_z);
                    glVertex3d(max_x,max_y,max_z);

                    glVertex3d(min_x,min_y,max_z);
                    glVertex3d(min_x,max_y,max_z);

                    glEnd();*/

                }
            }
            
        }
        glEnd();
    }
        

    if (bDrawdynamicinfo)
    {

        if(dynamic_map_points > 8)
        {
            /*glPushMatrix();
            #ifdef HAVE_GLES
                glMultMatrixf(Twc.m);
            #else
                glMultMatrixd(Twc.m);
            #endif

            glBegin(GL_QUADS);
            glColor3f(1.0f, 0.0f, 0.0f); 
            glVertex3d(-0.2,0.1,0);
            glVertex3d(0.2,0.1,0);
            glVertex3d(0.2,0.1,0);
            glVertex3d(0.2,0.1,0.4);
            glVertex3d(0.2,0.1,0.4);
            glVertex3d(-0.2,0.1,0.4);
            glVertex3d(-0.2,0.1,0.4);
            glVertex3d(-0.2,0.1,0);
            glEnd();
            glPopMatrix();*/
            glLineWidth(3);
            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            const float PI = 3.1415926f;
            float increment = 2.0f * PI / 50;
            for (float currAngle = 0.0f; currAngle <= 2.0f * PI; currAngle += increment)
            {
            glVertex3d(0.5 * cos(currAngle) + center_x, 0 , 0.5 * sin(currAngle) + center_z);
            }
            glEnd();
        }
        else
        {
            /*glPushMatrix();
            #ifdef HAVE_GLES
                glMultMatrixf(Twc.m);
            #else
                glMultMatrixd(Twc.m);
            #endif
            glBegin(GL_QUADS);
            glColor3f(1.0f, 1.0f, 0.0f); 
            glVertex3d(-0.2,0.1,0);
            glVertex3d(0.2,0.1,0);
            glVertex3d(0.2,0.1,0);
            glVertex3d(0.2,0.1,0.4);
            glVertex3d(0.2,0.1,0.4);
            glVertex3d(-0.2,0.1,0.4);
            glVertex3d(-0.2,0.1,0.4);
            glVertex3d(-0.2,0.1,0);
            glEnd();
            glPopMatrix();*/
            glLineWidth(3);
            glBegin(GL_LINES);
            glColor3f(1.0, 1.0, 0.0);
            const float PI = 3.1415926f;
            float increment = 2.0f * PI / 50;
            for (float currAngle = 0.0f; currAngle <= 2.0f * PI; currAngle += increment)
            {
            glVertex3d(0.5 * cos(currAngle) + center_x, 0 , 0.5 * sin(currAngle) + center_z);
            }
            glEnd();
        }
    }

    if (bDrawCurrentPoints)
    {
        // Define points
        glPointSize(5);
        glBegin(GL_POINTS);
        glColor3f(0.0, 0.0, 1.0);

        // All map points
        for(set<MapPoint*>::iterator sit=spCurrMPs.begin(), send=spCurrMPs.end(); sit!=send; sit++)
        {
            if ((*sit)->isBad())
                continue;
            cv::Mat pos = (*sit)->GetWorldPos();
            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        }
        glEnd();
    }

    
}
/****************** Semi dense mapping ********************
void MapDrawer::DrawSemiDense()
{

    const vector<KeyFrame*> &vpKf = mpMap->GetAllKeyFrames();
    if(vpKf.empty())return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,1.0,0.0);

    //for(size_t i = 0; i < vpKf.size();i=i+3)
    int draw_cnt(0);
    for(size_t i = 0; i < vpKf.size();++i)
    {
        KeyFrame* kf = vpKf[i];
        if(! kf->semidense_flag_) continue;
        draw_cnt ++;
        for(size_t y = 0; y< kf->im_.rows; y++)
          for(size_t x = 0; x< kf->im_.cols; x++)
        {

          Eigen::Vector3f Pw  (kf->SemiDensePointSets_.at<float>(y,3*x), kf->SemiDensePointSets_.at<float>(y,3*x+1), kf->SemiDensePointSets_.at<float>(y,3*x+2));
          //float z = Pw[2];
          if(Pw[2]>0)
          {
            float b = kf->rgb_.at<uchar>(y,3*x) / 255.0;
            float g = kf->rgb_.at<uchar>(y,3*x+1) / 255.0;
            float r = kf->rgb_.at<uchar>(y,3*x+2) / 255.0;
            glColor3f(r,g,b);
            glVertex3f( Pw[0],Pw[1],Pw[2]);
          }
        }
    }
    //if( draw_cnt>0) std::cout<<"Have Drawn : "<<draw_cnt<<"KeyFrame's semidense map "<<std::endl;
    glEnd();

}
****************************************************************/

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

} //namespace ORB_SLAM
