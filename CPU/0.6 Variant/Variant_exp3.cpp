/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-09
 * File Name   : Variant_exp3.cpp
 * Description : variant的部分成员函数的使用
 *****************************************************************/

#include <variant>
#include <iostream>
#include <string>
#include <typeinfo>

int main() {
    std::variant<int, float, std::string> var = "Hello";

    std::cout << "Current index: " << var.index() << std::endl;  // 输出 2

    if (var.index() == 0) {
        using T = std::variant_alternative<0, decltype(var)>::type;
        std::cout << "Current type: " << typeid(T).name() << " with value " << std::get<T>(var) << std::endl;
    } else if (var.index() == 1) {
        using T = std::variant_alternative<1, decltype(var)>::type;
        std::cout << "Current type: " << typeid(T).name() << " with value " << std::get<T>(var) << std::endl;
    } else if (var.index() == 2) {
        using T = std::variant_alternative<2, decltype(var)>::type;
        std::cout << "Current type: " << typeid(T).name() << " with value " << std::get<T>(var) << std::endl;
    }

    return 0;
}
// g++ -std=c++17 Variant_exp3.cpp -o Variant_exp3
/*********************************************************************************************
Current index: 2
Current type: NSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE with value Hello
**********************************************************************************************/