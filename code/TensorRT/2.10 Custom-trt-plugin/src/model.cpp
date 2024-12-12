#include <NvInfer.h>

#include <iostream>


#include "../inc/utils.hpp"
#include 


float input_5x5[] = {
    0.7576, 0.2793, 0.4031, 0.7347, 0.0293,
    0.7999, 0.3971, 0.7544, 0.5695, 0.4388,
    0.6387, 0.5247, 0.6826, 0.3051, 0.4635,
    0.4550, 0.5725, 0.4980, 0.9371, 0.6556,
    0.3138, 0.1980, 0.4162, 0.2843, 0.3398};

float input_1x5[] = {
    0.7576, 0.2793, 0.4031, 0.7347, 0.0293};

using namespace std;

class Logger : public nvinfer1::ILogger{
public: 
    virtual void log (Severity severity, const char* msg) noexcept override {
        string str;
        switch (severity) {
            case Severity::kINTERNAL_ERROR: str = RED   "[fatal]"  CLEAR; 
            case Severity::kERROR :         str = RED   "[fatal]"  CLEAR; 
            case Severity::kWARNING :       str = BLUE  "[fatal]"  CLEAR; 
            case Severity::kINFO :          str = YELLOW "[fatal]" CLEAR; 
            case Severity::kVERBOSE:        str = PURPLE "[verb]"  CLEAR;
        }
    }
};

