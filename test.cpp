#include <iostream>
using namespace std;

class Box
{
public:
    int getVloume(void){
        return l * w * h;
    }
    void setL(int ll){
        l = ll;
    }
    void setW(int ww){
        w = ww;
    }
    void setH(int hh){
        h = hh;
    }
    // 重载
    // Box operator + (const Box & b) const {
    //     Box box;
    //     box.l = this->l + b.l;
    //     box.w = this->w + b.w;
    //     box.h = this->h + b.h;
    //     return box;
    // }

private:
    int l, w, h;

};


int main(){
    Box box1;
    Box box2;
    Box box3;

    int V = 0;
    box1.setL(6);
    box1.setW(7);
    box1.setH(5);

    box2.setL(12);
    box2.setW(13);
    box2.setH(10);

    V = box1.getVloume();
    cout << "V1:" << V << endl;

    box3 = box1 + box2;
    V = box3.getVloume();
    cout << "V3:" << V << endl;
    
    return 0;

}