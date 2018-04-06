#ifndef SEMANTICMAPPOINT_HPP
#define SEMANTICMAPPOINT_HPP

#include <ros/ros.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <mutex>

class SemanticMapPoint{
public:
    SemanticMapPoint();
    ~SemanticMapPoint(){};
    SemanticMapPoint(int LabelID, float prob, cv::Point3f minpoint, cv::Point3f maxpoint){
        LabelID_ = LabelID;
        probability_ = prob;
        CornersMin_ = minpoint;
        CornersMax_ = maxpoint;
        Center_ = (minpoint + maxpoint) / 2;
    };
    void UpdateLabel(int LabelID, float prob){
        if(LabelID_ == LabelID){
            iObserved_ += 1;
            float alpha = probability_/(probability_ + prob);
            float logodd = log(prob/(1-prob))+log(probability_/(1-probability_));
            probability_ = 1/(1+exp(-logodd));
        }else{
            iObserved_ += 1;
            if(prob > probability_){
                LabelID_ =  LabelID;
                probability_ = prob;
            }
        }
    };
    void UpdateSize(cv::Point3f minPoint, cv::Point3f maxPoint){
        if(minPoint.x < CornersMin_.x){
            CornersMin_.x = minPoint.x;
        }
        if(minPoint.y < CornersMin_.y){
            CornersMin_.y = minPoint.y;
        }
        if(minPoint.z < CornersMin_.z){
            CornersMin_.z = minPoint.z;
        }
        if(maxPoint.x > CornersMax_.x){
            CornersMax_.x = maxPoint.x;
        }
        if(maxPoint.y > CornersMax_.y){
            CornersMax_.y = maxPoint.y;
        }
        if(maxPoint.z > CornersMax_.z){
            CornersMax_.z = maxPoint.z;
        }
        Center_ = CornersMin_/2 + CornersMax_/2;
    };
    void UpdateMapPoint(SemanticMapPoint* a){
        int LabelID = a->LabelID_;
        float prob = a->probability_;
        cv::Point3f minPoint = a->CornersMin_;
        cv::Point3f maxPoint = a->CornersMax_;
        if(LabelID == LabelID_){
            iObserved_ += 1;
            float alpha = probability_/(probability_ + prob);
            float logodd = log(prob/(1-prob))+log(probability_/(1-probability_));
            probability_ = 1/(1+exp(-logodd));
            CornersMin_ = alpha * CornersMin_ + (1 - alpha) * minPoint;
            CornersMax_ = alpha * CornersMax_ + (1 - alpha) * maxPoint;
            Center_ = CornersMin_/2 + CornersMax_/2;
        }else{
            iObserved_ += 1;
            if(prob > probability_){
                LabelID_ = LabelID;
                CornersMin_ = minPoint;
                CornersMax_ = maxPoint;
                Center_ = CornersMin_/2 + CornersMax_/2;
            }
        }
    }
    float getIoU(SemanticMapPoint comparedSemanticMP){
        cv::Point3f WLH = CornersMax_ - CornersMin_ + comparedSemanticMP.CornersMax_ - comparedSemanticMP.CornersMin_;
        // sum of with , length , height from two cube
        cv::Point3f fpDistance;
        fpDistance = Center_ - comparedSemanticMP.Center_;
        fpDistance.x = abs(fpDistance.x);
        fpDistance.y = abs(fpDistance.y);
        fpDistance.z = abs(fpDistance.z);
        fpDistance = fpDistance - WLH / 2;
        if(fpDistance.x >= 0 && fpDistance.y >= 0 && fpDistance.z >= 0){
            // No intersection
            return 0;
        }else{
            cv::Point3f wlh = CornersMax_ - CornersMin_;
            cv::Point3f cwlh = comparedSemanticMP.CornersMax_ - comparedSemanticMP.CornersMin_;
            float IVolume;
            cv::Point3f diffMin = CornersMin_ - comparedSemanticMP.CornersMin_;
            cv::Point3f diffMax = CornersMax_ - comparedSemanticMP.CornersMax_;
            float width,length,height;
            if(diffMin.x<0 && diffMin.y<0 && diffMin.z<0){
                if(diffMax.x>0){
                    width = cwlh.x;
                }else{
                    width = CornersMax_.x - comparedSemanticMP.CornersMin_.x;
                }
                if(diffMax.y>0){
                    length = cwlh.y;
                }else{
                    length = CornersMax_.y - comparedSemanticMP.CornersMin_.y;
                }
                if(diffMax.z>0){
                    height = cwlh.z;
                }else{
                    height = CornersMax_.z - comparedSemanticMP.CornersMin_.z;
                }
                
            }else if(diffMin.x>=0 && diffMin.y>=0 && diffMin.z>=0){
                if(diffMax.x<0){
                    width = wlh.x;
                }else{
                    width = comparedSemanticMP.CornersMax_.x - CornersMin_.x;
                }
                if(diffMax.y<0){
                    length = wlh.y;
                }else{
                    length = comparedSemanticMP.CornersMax_.y - CornersMin_.y;
                }
                if(diffMax.z<0){
                    height = wlh.z;
                }else{
                    height = comparedSemanticMP.CornersMax_.z - CornersMin_.z;
                }
                
            }else if(diffMin.x<=0 && diffMin.y>=0 && diffMin.z>=0){
                if(diffMax.x>0){
                    width = cwlh.x;
                }else{
                    width = CornersMax_.x - comparedSemanticMP.CornersMin_.x;
                }
                if(diffMax.y>0){
                    length = comparedSemanticMP.CornersMax_.y - CornersMin_.y;
                }else{
                    length = wlh.y;
                }
                if(diffMax.z>0){
                    height = CornersMax_.z - comparedSemanticMP.CornersMin_.z;
                }else{
                    height = wlh.z;
                }
                
            }else if(diffMin.x>=0 && diffMin.y<=0 && diffMin.z>=0){
                if(diffMax.x>0){
                    width = CornersMax_.x - comparedSemanticMP.CornersMin_.x;
                }else{
                    width = wlh.x;
                }
                if(diffMax.y>0){
                    length = cwlh.y;
                }else{
                    length = CornersMax_.y - comparedSemanticMP.CornersMin_.y;
                }
                if(diffMax.z>0){
                    height = CornersMax_.z - comparedSemanticMP.CornersMin_.z;
                }else{
                    height = wlh.z;
                }
                
            }else if(diffMin.x>=0 && diffMin.y>=0 && diffMin.z<=0){
                if(diffMax.x>0){
                    width = CornersMax_.x - comparedSemanticMP.CornersMin_.x;
                }else{
                    width = wlh.x;
                }
                if(diffMax.y>0){
                    length = comparedSemanticMP.CornersMax_.y - CornersMin_.y;
                }else{
                    length = wlh.y;
                }
                if(diffMax.z>0){
                    height = cwlh.z;
                }else{
                    height = CornersMax_.z - comparedSemanticMP.CornersMin_.z;
                }
                
            }else if(diffMin.x>=0 && diffMin.y<=0 && diffMin.z<=0){
                if(diffMax.x>0){
                    width = comparedSemanticMP.CornersMax_.x - CornersMin_.x;
                }else{
                    width = wlh.x;
                }
                if(diffMax.y>0){
                    length = cwlh.y;
                }else{
                    length = comparedSemanticMP.CornersMax_.y - CornersMin_.y;
                }
                if(diffMax.z>0){
                    height = cwlh.z;
                }else{
                    height = comparedSemanticMP.CornersMax_.z - CornersMin_.z;
                }
            }else if(diffMin.x<=0 && diffMin.y>=0 && diffMin.z<=0){
                if(diffMax.x>0){
                    width = cwlh.x;
                }else{
                    width = comparedSemanticMP.CornersMax_.x - CornersMin_.x;
                }
                if(diffMax.y>0){
                    length = comparedSemanticMP.CornersMax_.y - CornersMin_.y;
                }else{
                    length = wlh.y;
                }
                if(diffMax.z>0){
                    height = cwlh.z;
                }else{
                    height = comparedSemanticMP.CornersMax_.z - CornersMin_.z;
                }
            }else if(diffMin.x<=0 && diffMin.y<=0 && diffMin.z>=0){
                if(diffMax.x>0){
                    width = cwlh.x;
                }else{
                    width = comparedSemanticMP.CornersMax_.x - CornersMin_.x;
                }
                if(diffMax.y>0){
                    length = cwlh.y;
                }else{
                    length = comparedSemanticMP.CornersMax_.y - CornersMin_.y;
                }
                if(diffMax.z>0){
                    height = comparedSemanticMP.CornersMax_.z - CornersMin_.z;
                }else{
                    height = wlh.z;
                }
            }
            IVolume = width * length * height;
            float UVolume = wlh.x*wlh.y*wlh.z + cwlh.x*cwlh.y*cwlh.z - IVolume;
            float fIoU = IVolume / UVolume;
            if(fIoU<0){
                std::cout<<"min"<<diffMin<<std::endl;
                std::cout<<"max"<<diffMax<<std::endl;
                std::cout<<width<<","<<length<<","<<height<<std::endl;
            }
            return fIoU;
        }
    }
    
public:
    long unsigned int uId;
    static long unsigned int uNextId;
    int LabelID_;
    float probability_;
    cv::Point3f CornersMin_;
    cv::Point3f CornersMax_;
    cv::Point3f Center_;
    // min,min,min - max,max,max
    //   5 -- 6
    //  /    /|
    // 1 -- 2 8
    // |    |/
    // 3 -- 4
private:
    // Observe counters
    int iObserved_;

};

#endif