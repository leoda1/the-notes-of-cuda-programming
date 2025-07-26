#include <iostream>

using namespace std;

void printValue(int &i) {
    cout << "l-value: " << i << endl;
}

void printValue(int &&i) {
    cout << "r-value: " << i << endl;
}

void forward(int && k) {
    printValue(k);
}

int main() {
    int value = 500;
    printValue(value);
    printValue(500);
    forward(1000);
    return 0;
}

/******************************************************************
l-value: 500
r-value: 500
l-value: 1000
 *****************************************************************/