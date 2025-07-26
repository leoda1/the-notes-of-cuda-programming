/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-09
 * File Name   : Variant_exp2.cpp
 * Description : variant的第二个简单例子
 *****************************************************************/
#include <iostream>
#include <variant>
#include <string>

using namespace std;

int main() {
    variant<int, string> myIntStr;
    myIntStr = 100;
    myIntStr = "leoda";

    try {
        if (holds_alternative<int>(myIntStr)) {
            cout << "value as int is : " << get<int>(myIntStr) << endl;
        } else if (holds_alternative<string>(myIntStr)) {
            cout << "value as string is : " << get<string>(myIntStr) << endl;
        } else {
            cout << "error"<< endl;
        }
    } catch (bad_variant_access& e){
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}

// g++ -std=c++17 Variant_exp2.cpp -o Variant_exp2
/******************************************************************
value as string is : leoda
*******************************************************************/