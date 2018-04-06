#ifndef SEMANTIC_HPP
#define SEMANTIC_HPP

#include <ros/ros.h>
#include "SemanticMapPoint.hpp"
#include "RegistedMapPoint.hpp"
#include <set>
#include <mutex>

using namespace std;

class SemanticMap{
public:
    SemanticMap(){};
    void addSemanticMapPoint(SemanticMapPoint* pSMP){
        unique_lock<mutex> lock(mMutexMap);
        pair<set<SemanticMapPoint*>::iterator,bool> pr;
        pr = sSemanticMapPoints.insert(pSMP);
        std::cout<<"insert semantic point"<<pr.second<<std::endl;
        // std::cout<<"sSemanticMapPoints size:" << sSemanticMapPoints.size()<<std::endl;
    };
    void addRegistedMapPoint(RegistedMapPoint* pRMP){
        unique_lock<mutex> lock(mMutexMap);
        pair<set<RegistedMapPoint*>::iterator,bool> pr;
        pr = sRegistedMapPoints.insert(pRMP);
        std::cout<<"insert registed point"<<pr.second<<std::endl;
        std::cout<<"sRegistedMapPoints size:" << sSemanticMapPoints.size()<<std::endl;
    };
    void eraseSemanticMapPoint(SemanticMapPoint* pSMP){
        unique_lock<mutex> lock(mMutexMap);
        sSemanticMapPoints.erase(pSMP);
    };
    std::vector<SemanticMapPoint*> getAllSemanticMapPoints(){
        unique_lock<mutex> lock(mMutexMap);
        return std::vector<SemanticMapPoint*>(sSemanticMapPoints.begin(),sSemanticMapPoints.end());
    };
    std::vector<RegistedMapPoint*> getAllRegistedMapPoints(){
        unique_lock<mutex> lock(mMutexMap);
        return std::vector<RegistedMapPoint*>(sRegistedMapPoints.begin(),sRegistedMapPoints.end());
    }
    ~SemanticMap(){}; 

protected:
    std::set<SemanticMapPoint*> sSemanticMapPoints;
    std::set<RegistedMapPoint*> sRegistedMapPoints;
    std::mutex mMutexMap;
};

#endif