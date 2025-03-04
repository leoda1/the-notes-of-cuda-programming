#include <iostream>

int main() {
    int&& a1 = 5;
    auto&& bb = a1; //a1为右值引用，推导出的bb为左值引用类型
    auto&& bb1 = 5; //5为右值，推导出的bb1为右值引用类型

    int a2 = 5;
    int &a3 = a2;
    auto&& cc = a3; //a3为左值引用，推导出的cc为左值引用类型
    auto&& cc1 = a2;//a2为左值，推导出的cc1为左值引用类型

    const int& s1 = 100; 
    const int&& s2 = 100;
    auto&& dd = s1; //s1为常量左值引用，推导出的dd为常量左值引用类型
    auto&& ee = s2; //s2为常量右值引用，推导出的ee为常量左值引用类型

    const auto&& x = 5;//x为右值引用，不需要推导，只能通过右值初始化

    // 输出验证
    std::cout << "bb (int&): " << bb << std::endl;
    std::cout << "bb1 (int&&): " << bb1 << std::endl;
    std::cout << "cc (int&): " << cc << std::endl;
    std::cout << "cc1 (int&): " << cc1 << std::endl;
    std::cout << "dd (const int&): " << dd << std::endl;
    std::cout << "ee (const int&): " << ee << std::endl;
    std::cout << "x (const int&&): " << x << std::endl;

    return 0;
}
/******************************************************************
bb (int&): 5
bb1 (int&&): 5
cc (int&): 5
cc1 (int&): 5
dd (const int&): 100
ee (const int&): 100
x (const int&&): 5
 *****************************************************************/