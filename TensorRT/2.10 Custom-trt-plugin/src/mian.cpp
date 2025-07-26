#include <iostream>
#include <memory>

using namespace std;

int main(int argc, char const *argv[]) {
    Model model();
    if (!model.build()) {
        LOGE("fail in building model");
        return 0;
    }
    if (!model.infer()) {
        LOGE("fail in infering model");
        return 0;
    }
    return 0;
}