#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <iostream>
using namespace std;

class Model{
public:
    enum precision {
        FP32,
        FP16,
        INT8
    };

    Model(string onnxpath, precision prec);
    bool build();
    bool infer();

private:
     
};



#endif // __MODEL_HPP__