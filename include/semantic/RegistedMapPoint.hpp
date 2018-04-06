#ifndef REGISTEDMAPPOINT_HPP
#define REGISTEDMAPPOINT_HPP

#include <ros/ros.h>
#include "SemanticMapPoint.hpp"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <mutex>

class RegistedMapPoint{
public:
    RegistedMapPoint();
    ~RegistedMapPoint(){};
    RegistedMapPoint(cv::Point3f pt, SemanticMapPoint* SMP){
        pfMapPoint = pt;
        ptrSemanticMP = SMP;
    };
    
public:
    cv::Point3f pfMapPoint;
    SemanticMapPoint* ptrSemanticMP;
private:

};

#endif