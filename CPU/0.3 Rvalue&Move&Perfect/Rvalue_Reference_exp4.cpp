#include <iostream>
#include <vector>
#include <string.h>
using namespace std;

class mystr {
private:
    char *data;
    size_t len;
    void copy_data(const char *s) {
        data = new char[len + 1];
        memcpy(data, s, len);
        data[len] = '\0';
    }

public:
    mystr() {
        data = NULL;
        len = 0;
    }
    mystr(const char *p) {
        len = strlen(p);
        copy_data(p);
    }
    // mystr(const mystr& str) {
    //     len = str.len;
    //     copy_data(str.data);
    //     cout << "copy constructor is called! source :" << str.data << endl;
    // }
    mystr(mystr&& str) noexcept {
        cout << "move constructor is called! source: " << str.data << endl;
        data = str.data;
        len = str.len;
        str.data = nullptr;
        str.len = 0;
    }
    mystr& operator=(mystr&& str) noexcept {
        cout << "move assignment is called! source: " << str.data << endl;
        if (this != &str) {
            delete[] data;
            data = str.data;
            len = str.len;
            str.data = nullptr;
            str.len = 0;
        }
        return *this;
    }
    // mystr& operator=(const mystr& str) {
    //     if (this != &str) {
    //         delete[] data;  // 释放旧内存，避免泄漏
    //         len = str.len;
    //         copy_data(str.data);
    //     }
    //     cout << "copy assignment is called! source :" << str.data << endl;
    //     return *this;
    // }
    virtual ~mystr() {
        if (data) free(data);
    }
};

void test () {
    mystr str;
    str = mystr("Hello World");
    vector<mystr> vec;
    vec.emplace_back(mystr("bro"));
}

int main() {
    test();
    return 0;
}
/******************************************************************
copy assignment is called! source :Hello World
copy constructor is called! source :bro
 *****************************************************************/

/******************************************************************
move assignment is called! source: Hello World
move constructor is called! source: bro
 *****************************************************************/