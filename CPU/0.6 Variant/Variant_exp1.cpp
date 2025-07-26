/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-09
 * File Name   : Variant_exp1.cpp
 * Description : 使用variant的简单例子
 *****************************************************************/

#include <iostream>
#include <variant>
int main() {
    std::variant<int, double> myVariant;

    myVariant = 42; /* Store an int */

    std::cout << std::get<double>(myVariant) << std::endl;

    return 0;
}

// g++ -std=c++17 Variant_exp1.cpp -o Variant_exp1
/*
terminating due to uncaught exception of type std::bad_variant_access: bad_variant_access
zsh: abort
*/
