#include <ros/ros.h>
#include <map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>   
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/miniflann.hpp>

#include "darknet_ros_msgs/BoundingBoxes.h"
#include "darknet_ros_msgs/BoundingBox.h"
#include "orb_slam2/MapKeyPoints.h"

#include "semantic/SemanticMapPoint.hpp"
#include "semantic/Map.hpp"

using namespace message_filters;
using namespace std;

#define DistThreshold 0.1

bool init_status = false;
std::map<ros::Time, std::vector<geometry_msgs::Point>> KPstack;
std::vector<std::vector<geometry_msgs::Point>> MPstack;

std::map<string, int> Label_Id;
std::vector<std::vector<float>> IdColor;

ros::Publisher image_pub;
ros::Publisher Map_pub;
ros::Publisher RegistedPoints_pub;
ros::Publisher ORBMapPoints_pub;
SemanticMap* mpMap = new SemanticMap();


bool inBox(cv::Point2f point, darknet_ros_msgs::BoundingBox box){
  if(point.x < box.xmin || point.x > box.xmax || point.y < box.ymin || point.y > box.ymax){
    return false;
  }else{
    return true;
  }
}
float Cubic(float a0, float a1, float a2, float a3, float x){
  return a0 + a1 * x + a2 * x * x + a3 * x * x * x;
}
std::vector<float> color(float value, int v_min = 0, int v_max = 80, float v_cnt = 40){
  float x, r, g, b;
  std::vector<float> rgb;
  if (value < v_cnt){
    x = 0.5*(value-v_min)/(v_cnt-v_min);
  }else{
    x = 0.5*(value-v_cnt)/(v_max-v_cnt)+0.5;
  }
  if(x<.137254902){
    r = 0;
  }else if(x<.309803922){
    r = Cubic( -1.782709, 22.937070,  -86.635448, 109.222171, x);
  }else if(x<.623529412){
    r = Cubic( -0.765706,  3.908358,   -2.385836,   0.911563, x);
  }
  else if(x<.741176471){
    r = Cubic(-18.069920, 81.220268, -115.432390,  54.731649, x);
  }
  else if(x<.917647059){
    r = 1;
  }else{
    r = Cubic( 0.688335, 0.804310, -0.553402, 0.051166, x);
  }

  if(x<.611764706){
    g = Cubic(  0.221600,  2.096260,   -2.989336,   1.865388, x);
  }else{
    g = Cubic(  2.304104, -4.665514,    5.405245,  -2.842239, x);
  }

  if(x<.345098039){
    b = Cubic(  0.964755, -3.046139,    3.252455,  -6.730486, x);
  }else if(x<.623529412){
    b = Cubic(  0.034808, -0.485577,    2.123163,  -1.570901, x);
  }else if(x<.964705882){
    b = Cubic( -0.692383,  4.166389,   -6.037159,   2.550446, x);
  }else{
    b = 0;
  }
  if(r<0)r = 0;
  if(r>1)r = 1;
  if(g<0)g = 0;
  if(g>1)g = 1;
  if(b<0)b = 0;
  if(b>1)b = 1;
  
  rgb.push_back(r);
  rgb.push_back(g);
  rgb.push_back(b);
  return rgb;
}

float getDistance2(cv::Point3f p1, cv::Point3f p2){
  float x = p1.x-p2.x;
  float y = p1.y-p2.y;
  float z = p1.z-p2.z;
  return x*x + y*y + z*z;
}

void drawSemanticMap(){
  std::vector<SemanticMapPoint*> mpSemanticMPs = mpMap -> getAllSemanticMapPoints();
  std::vector<RegistedMapPoint*> mpRegistedMPs = mpMap -> getAllRegistedMapPoints();
  auto map_iter = mpSemanticMPs.cbegin();
  auto registedPt_iter = mpRegistedMPs.cbegin();
  visualization_msgs::Marker DisplayRegistedPoints;
  visualization_msgs::MarkerArray CubeArray;
  int cube_id = 0;
  ROS_INFO_STREAM("Semantic Map Points size:"<<mpSemanticMPs.size());
  while(map_iter != mpSemanticMPs.cend()){
    cv::Point3f DoubleScale = (*map_iter) -> CornersMax_ - (*map_iter) -> CornersMin_;
    // std::cout<<(*map_iter) -> CornersMax_ - (*map_iter) ->CornersMin_<<std::endl;
    visualization_msgs::Marker cube;
    cube.header.frame_id = "/world";
    cube.header.stamp = ros::Time::now();
    cube.ns = "SemanticMapPoints";
    cube.action = visualization_msgs::Marker::ADD;
    cube.pose.orientation.w = 1.0;
    cube.id = cube_id++;
    // ok
    cube.lifetime = ros::Duration();
    cube.type = visualization_msgs::Marker::CUBE;
    cube.color.a = (*map_iter)->probability_;
    cube.color.r = IdColor[(*map_iter)->LabelID_][0];
    cube.color.g = IdColor[(*map_iter)->LabelID_][1];
    cube.color.b = IdColor[(*map_iter)->LabelID_][2];
    cube.scale.x = DoubleScale.x / 2;
    cube.scale.y = DoubleScale.y / 2;
    cube.scale.z = DoubleScale.z / 2;
    cube.pose.position.x = (*map_iter) -> Center_.x;
    cube.pose.position.y = (*map_iter) -> Center_.y;
    cube.pose.position.z = (*map_iter) -> Center_.z;
    CubeArray.markers.push_back(cube);
    ++map_iter;
  }
  ROS_INFO_STREAM("Registed Points size:"<<mpRegistedMPs.size());
  while(registedPt_iter != mpRegistedMPs.cend()){
    DisplayRegistedPoints.header.frame_id = "/world";
    DisplayRegistedPoints.header.stamp = ros::Time::now();
    DisplayRegistedPoints.ns = "RegistedMapPoints";
    DisplayRegistedPoints.action = visualization_msgs::Marker::ADD;
    DisplayRegistedPoints.pose.orientation.w = 1.0;
    DisplayRegistedPoints.id = 0;
    // ok
    DisplayRegistedPoints.lifetime = ros::Duration();
    DisplayRegistedPoints.type = visualization_msgs::Marker::POINTS;
    DisplayRegistedPoints.color.a = 1.0;
    DisplayRegistedPoints.color.r = IdColor[(*registedPt_iter)->ptrSemanticMP->LabelID_][0];
    DisplayRegistedPoints.color.g = IdColor[(*registedPt_iter)->ptrSemanticMP->LabelID_][1];
    DisplayRegistedPoints.color.b = IdColor[(*registedPt_iter)->ptrSemanticMP->LabelID_][2];
    DisplayRegistedPoints.scale.x = 0.1;
    DisplayRegistedPoints.scale.y = 0.1;
    DisplayRegistedPoints.scale.z = 0.1;
    geometry_msgs::Point p;
    p.x = (*registedPt_iter) -> pfMapPoint.x;
    p.y = (*registedPt_iter) -> pfMapPoint.y;
    p.z = (*registedPt_iter) -> pfMapPoint.z;
    DisplayRegistedPoints.points.push_back(p);
    ++registedPt_iter;
  }
  // Map_pub.publish(CubeArray);
  // RegistedPoints_pub.publish(DisplayRegistedPoints);
}

void BBoxCallback(const darknet_ros_msgs::BoundingBoxes& inputBoxes){
  // std::cout<<"get BoundingBoxes";
  if(init_status){
    ros::Time boxtime = inputBoxes.frametime;
    unsigned int mp_i = 0;
    auto kp_iter = KPstack.cbegin();
    std::vector<SemanticMapPoint*> mpSemanticMPs = mpMap -> getAllSemanticMapPoints();
    std::vector<RegistedMapPoint*> mpRegistedMPs = mpMap -> getAllRegistedMapPoints();
    ROS_INFO_STREAM("RegistedMPs size"<<mpRegistedMPs.size());
    cv::Mat RegMP_mat;
    if(!mpRegistedMPs.empty()){
      // ROS_INFO_STREAM("RegistedMPs");
      // build k-d tree for search
      RegMP_mat = cv::Mat(mpRegistedMPs.size(),3,CV_32FC1);
      for(size_t i = 0; i < mpRegistedMPs.size(); i++){
        RegMP_mat.at<float>(i,0) = mpRegistedMPs[i] -> pfMapPoint.x;
        RegMP_mat.at<float>(i,1) = mpRegistedMPs[i] -> pfMapPoint.y;
        RegMP_mat.at<float>(i,2) = mpRegistedMPs[i] -> pfMapPoint.z;
      }
    }else{
      RegMP_mat = cv::Mat::zeros(1,3,CV_32F);
    }
    cvflann::KDTreeIndexParams indexParams(4);//（此参数用来设置构建的数据结构，此处选择K-d树）
    cv::flann::GenericIndex<cvflann::L2<float>> kdtree(RegMP_mat, indexParams);
    int num_k = 1; // number of nearst neighbours
    cv::Mat index(mpRegistedMPs.size(),num_k,CV_32S);
    cv::Mat dist_result(mpRegistedMPs.size(),num_k,CV_32F);
    cv::Mat source = RegMP_mat;
    
    // traverse key points & find corresponding map points in boxes
    while(kp_iter != KPstack.cend()){
      ros::Duration timediff = boxtime - kp_iter->first;
      if(timediff.toSec()<0.5){//match data
        // keypoints
        std::vector<geometry_msgs::Point> ORB_keypoints = kp_iter->second; 
        // BoundingBoxes
        std::vector<darknet_ros_msgs::BoundingBox> boxes = inputBoxes.boundingBoxes;
        // Map Points corresponding to a box
        std::vector<geometry_msgs::Point> BoxedMPs;
        cv::Point3f minPoint;
        cv::Point3f maxPoint;
        visualization_msgs::Marker ORBMapPoints;
        ORBMapPoints.header.frame_id = "/world";
        ORBMapPoints.header.stamp = ros::Time::now();
        ORBMapPoints.ns = "RegistedMapPoints";
        ORBMapPoints.action = visualization_msgs::Marker::ADD;
        ORBMapPoints.pose.orientation.w = 1.0;
        ORBMapPoints.id = 0;
        ORBMapPoints.lifetime = ros::Duration();
        ORBMapPoints.type = visualization_msgs::Marker::POINTS;
        ORBMapPoints.color.a = 1.0;
        ORBMapPoints.color.r = 1.0;
        ORBMapPoints.color.g = 1.0;
        ORBMapPoints.color.b = 1.0;
        ORBMapPoints.scale.x = 0.1;
        ORBMapPoints.scale.y = 0.1;
        ORBMapPoints.scale.z = 0.1;
        ROS_INFO_STREAM("box size: "<<boxes.size());
        for(size_t j = 0; j < boxes.size(); j++){
          for(size_t i = 0; i < ORB_keypoints.size(); i++){
            cv::Point2f pt;
            pt.x = ORB_keypoints[i].x;
            pt.y = ORB_keypoints[i].y;
            if(inBox(pt, boxes[j])){
              ROS_INFO_STREAM_ONCE("point in box");
              BoxedMPs.push_back(MPstack[mp_i][i]);
              minPoint.x = minPoint.x < MPstack[mp_i][i].x ? minPoint.x : MPstack[mp_i][i].x;
              minPoint.y = minPoint.y < MPstack[mp_i][i].y ? minPoint.y : MPstack[mp_i][i].y;
              minPoint.z = minPoint.z < MPstack[mp_i][i].z ? minPoint.z : MPstack[mp_i][i].z;
              maxPoint.x = maxPoint.x > MPstack[mp_i][i].x ? maxPoint.x : MPstack[mp_i][i].x;
              maxPoint.y = maxPoint.y > MPstack[mp_i][i].y ? maxPoint.y : MPstack[mp_i][i].y;
              maxPoint.z = maxPoint.z > MPstack[mp_i][i].z ? maxPoint.z : MPstack[mp_i][i].z;
            }
            /* record map point for plot */
            if(j == 0){
              geometry_msgs::Point p;
              p.x = MPstack[mp_i][i].x;
              p.y = MPstack[mp_i][i].y;
              p.z = MPstack[mp_i][i].z;
              ORBMapPoints.points.push_back(p);
            }
          }
          
          if(mpSemanticMPs.empty()){
            // nothing in Map, and add a new map point
            ROS_INFO_STREAM("no map point");
            if(!BoxedMPs.empty()){
              ROS_INFO_STREAM_ONCE("what the hell"<<minPoint<<maxPoint);
              SemanticMapPoint* a = new SemanticMapPoint(Label_Id[boxes[j].Class], boxes[j].probability, minPoint, maxPoint);
              mpMap -> addSemanticMapPoint(a);
              ROS_INFO_STREAM("BoxedMPs size"<<BoxedMPs.size());
              for(size_t i = 0; i < BoxedMPs.size(); i++){
                cv::Point3f RMPpt(BoxedMPs[i].x,BoxedMPs[i].y,BoxedMPs[i].z);
                RegistedMapPoint* b = new RegistedMapPoint(RMPpt, a);
                mpMap -> addRegistedMapPoint(b);
              }
            }
          }else{
            // check whether the points are in map
            cv::Mat ser(BoxedMPs.size(),3,CV_32FC1);
            for(size_t i = 0; i < BoxedMPs.size(); i++){
              ser.at<float>(i,0) = BoxedMPs[i].x;
              ser.at<float>(i,1) = BoxedMPs[i].y;
              ser.at<float>(i,2) = BoxedMPs[i].z;
            }
            
            // flann get what's old
            int num_k = 1; // number of nearst neighbours
            cv::Mat nearest_index_result(BoxedMPs.size(),num_k,CV_32S);
            cv::Mat nearest_dist_result(BoxedMPs.size(),num_k,CV_32F);
            ROS_INFO_STREAM("start to flann");
            if(!mpRegistedMPs.empty()){
              kdtree.knnSearch(ser, nearest_index_result, nearest_dist_result, num_k, cvflann::SearchParams(64));
              ROS_INFO_STREAM("flann finished"<<nearest_index_result.size());
              // check if enough points are in map
              int DistCount = 0;
              std::vector<cv::Point3f> NotFoundPoints;
              set<SemanticMapPoint*> FoundSemMapPoints;
              for(size_t i = 0; i < BoxedMPs.size(); i++){
                if(nearest_dist_result.at<float>(i,0)<0.01){
                  FoundSemMapPoints.insert(mpRegistedMPs[nearest_index_result.at<int>(i,0)]->ptrSemanticMP);
                  DistCount++;
                }else{
                  cv::Point3f NotFoundPt(BoxedMPs[i].x,BoxedMPs[i].y,BoxedMPs[i].z);
                  NotFoundPoints.push_back(NotFoundPt);
                }
              }
              ROS_INFO_STREAM("NotFoundPoints"<<NotFoundPoints.size());
              ROS_INFO_STREAM("FoundSemMapPoints"<<FoundSemMapPoints.size());
              ROS_INFO_STREAM("DistCount size: "<<DistCount<<", BoxedMP size: "<<BoxedMPs.size());
              if(DistCount*1.0/BoxedMPs.size() > 0.5){
                ROS_INFO_STREAM("Enough Points");
                // enough points, update
                auto FoundSemMapPoints_iter = FoundSemMapPoints.begin();
                while(FoundSemMapPoints_iter != FoundSemMapPoints.end()){
                  (*FoundSemMapPoints_iter) -> UpdateLabel(Label_Id[boxes[j].Class], boxes[j].probability);
                  FoundSemMapPoints_iter++;
                }
                for(size_t NFPi = 0; NFPi < NotFoundPoints.size(); NFPi++){
                  cv::Point3f RMPpt(NotFoundPoints[NFPi].x,NotFoundPoints[NFPi].y,NotFoundPoints[NFPi].z);
                  RegistedMapPoint* b = new RegistedMapPoint(RMPpt, *(FoundSemMapPoints.begin()));
                  mpMap -> addRegistedMapPoint(b);
                }
                (*FoundSemMapPoints.begin())->UpdateSize(minPoint, maxPoint);
              }else{
                // not enough points, insert a new one
                SemanticMapPoint* a = new SemanticMapPoint(Label_Id[boxes[j].Class], boxes[j].probability, minPoint, maxPoint);
                mpMap -> addSemanticMapPoint(a);
                ROS_INFO_STREAM("Not Enough Points, add a new semantic map point");
                for(size_t NFPi = 0; NFPi < NotFoundPoints.size(); NFPi++){
                  cv::Point3f RMPpt(NotFoundPoints[NFPi].x,NotFoundPoints[NFPi].y,NotFoundPoints[NFPi].z);
                  ROS_INFO_STREAM("Not Enough Points, add a registed point"<<RMPpt);
                  RegistedMapPoint* b = new RegistedMapPoint(RMPpt, a);
                  mpMap -> addRegistedMapPoint(b);
                }
              }
              NotFoundPoints.clear();
              FoundSemMapPoints.clear();
            }

            /* IoU check for 
            auto map_iter = mpSemanticMPs.begin();
            auto minDistIter = map_iter;
            // float minDist2 = getDistance2(aCenter, (*map_iter) -> Center_);
            float maxIoU = (*map_iter) -> getIoU(*a);
            float IoU;
            while(map_iter != mpSemanticMPs.end()){
              // float MPDist2 = getDistance2(aCenter, (*map_iter) -> Center_);
              IoU = (*minDistIter) -> getIoU(*a);
              if(IoU >= maxIoU){
                minDistIter = map_iter;
                maxIoU = IoU;
              }
              map_iter++;
            }
            ROS_INFO_STREAM("IoU"<<IoU);
            if(IoU>0.5){
              (*minDistIter) -> UpdateMapPoint(a);
            }else{
              mpMap -> addSemanticMapPoint(a);
            } */
          
          }
          BoxedMPs.clear();
          
        }
        ORBMapPoints_pub.publish(ORBMapPoints);
        /* map point */


        KPstack.erase(KPstack.cbegin(),kp_iter);        
        MPstack.erase(MPstack.cbegin(),MPstack.cbegin()+mp_i);
        break;
      }
      ++kp_iter;
      ++mp_i;
    }
    /* draw Semantic map */
    drawSemanticMap();
  }
}


void pointsCallback(const orb_slam2::MapKeyPoints& inputPoints){
  if(!inputPoints.mappoints.empty()){
    init_status = true;
    ROS_INFO_STREAM_ONCE("inited");
    KPstack.insert({inputPoints.frametime,inputPoints.keypoints});
    MPstack.push_back(inputPoints.mappoints);
  }
}

void callback(const sensor_msgs::ImageConstPtr& img, const darknet_ros_msgs::BoundingBoxesConstPtr& inputBoxes){
  ROS_INFO_STREAM_ONCE("WTF");
  if(init_status){
    ros::Time boxtime = inputBoxes->frametime;
    auto map_it = KPstack.cbegin();
    while(map_it != KPstack.cend()){
      ros::Duration timediff = boxtime - map_it->first;
      // std::cout << timediff.toSec() << std::endl;
      if(timediff.toSec()<0.5){//key
        std::vector<geometry_msgs::Point> ORB_keypoints = map_it->second; //keypoints
        std::vector<darknet_ros_msgs::BoundingBox> boxes = inputBoxes->boundingBoxes;
        cv_bridge::CvImagePtr cv_ptr;
        try{
          cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        }catch (cv_bridge::Exception& e){
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;
        }
        for(size_t i = 0; i < ORB_keypoints.size(); i++){
          cv::Point2f pt;
          pt.x = ORB_keypoints[i].x;
          pt.y = ORB_keypoints[i].y;
          for(size_t j = 0; j < boxes.size(); j++){
            if(inBox(pt, boxes[j])){
              ROS_INFO_STREAM_ONCE("point in box");
              cv::circle(cv_ptr->image, pt, 5, CV_RGB(0,255,0));
            }
          }
        }
        KPstack.erase(KPstack.cbegin(),map_it);
        image_pub.publish(cv_ptr->toImageMsg());
        break;
      } 
      ++map_it;
    }
  }
}

int main (int argc, char** argv)
{
  float scale_linear = 0.5;
  float scale_angular = 0.2;
  // Initialize ROS
  ros::init (argc, argv, "SemanticMap");
  ros::NodeHandle nh;
  std::vector<string> classLabels;
  
  /******************** Parameter ********************/
  if(nh.getParam("/darknet_ros/yolo_model/detection_classes/names", classLabels)){
    ROS_INFO_STREAM("parameter loaded "<<classLabels.size());
  }else{
    ROS_ERROR("fail to load parameter");
  };
                    // std::vector<std::string>(0));
  for(size_t i = 0; i < classLabels.size(); i++){
    Label_Id.insert({classLabels[i],i});
    IdColor.push_back(color(i,0,classLabels.size(),classLabels.size()/2));
  }
  /******************** Subscriber topic ********************/
  ros::Subscriber KP_sub = nh.subscribe ("MapKeyPoints", 10, pointsCallback);
  // ros::Subscriber BBox_sub = nh.subscribe ("Box", 10, BBoxCallback);
  
  /******************** Publish topic ********************/
  image_pub = nh.advertise<sensor_msgs::Image>("KBoxImage",1,true);
  Map_pub = nh.advertise<visualization_msgs::MarkerArray>("SemanticMap",1,true);
  RegistedPoints_pub = nh.advertise<visualization_msgs::Marker>("RegistedPoints",1,true);
  ORBMapPoints_pub = nh.advertise<visualization_msgs::Marker>("ORBMapPoints",1,true);
  /******************** message_filters ********************/
  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh,"/darknet_ros/detection_image",10);
  message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> Box_sub(nh,"boundingboxes",10);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, Box_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  /****************** message_filters end ******************/
  // Create a ROS publisher
  
  ros::spin();
  // ros::Rate loop_rate(10); //循环频率控制对象，控制while循环一秒循环10次
  // while (ros::ok()) //ros::ok()只有当用户按下Crtl+C按键时，才会返回false，循环终止
  // {
  //   ros::spinOnce(); // 执行这条函数的意义在于，让主线程暂停，处理来自订阅者的请求
  //   loop_rate.sleep(); // 执行线程休眠
  // }
}


/********************Bounding Box********************/
// header:seq
//   stamp:
//    secs:
//    nsecs:
//   frame_id:detection
// BoundingBoxes:
//   Class:
//   probability:
//   xmin:
//   ymin:
//   xmax:
//   ymax:
