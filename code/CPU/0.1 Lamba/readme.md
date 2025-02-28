## 1 基本语法

1. **[捕获列表] (参数列表) -> 返回类型{函数体}**
    
    **例如：**`auto Add = [](int a, int b) -> int {return a + b;};` 
    
    <aside>
    👉🏻
    
    一般情况下，编译器会自动推断出**返回类型**，所以可以不指定。
    
    </aside>
    
2. **可以简化为： [捕获列表] (参数列表) {函数体}**
    
    **例如：**`auto Add = [](int a, int b) {return a + b;};`   
    
    <aside>
    ⚠️
    
    在C++11之后，如果Lamba表达式的函数体内有多个return语句且返回类型不一致，编译器无法自动推断的时候，此时必须显式的指定返回类型。
    
    </aside>
    
    **范例1 ：**
    
    ```cpp
    // 由于函数体内有多个 return 语句，编译器无法自动推断出返回类型，
    // 因此我们使用 -> string 明确指定了返回类型。
    #include <iostream>
    #include <string>
    using namespace std;
    
    int main() {
        auto example = [](int x) -> string {
            if (x > 0) return "Positive";
            else if (x == 0) return "Negative";
            else return "None";
        };
    
        cout << example(5) << endl;
        cout << example(0) << endl;
        cout << example(-15) << endl;
    
        return 0;
    }
    ```
    
    **范例2：**
    
    当出现返回的类型有多种情况的时候：
    
    ```cpp
    #include <iostream>
    #include <variant>
    #include <string>
    using namespace std;
    
    int main() {
        auto example = [](int x) -> variant<int, string> {
            if (x > 0) return x;
            else return "None";
        };
    
        auto result1 = example(5);
        auto result2 = example(-5);
    
        if (holds_alternative<int>(result1)) {
            cout << "Result1: " << get<int>(result1) << endl;
        } else {
            cout << "Result1: " << get<string>(result1) << endl;
        }
    
        if (holds_alternative<int>(result2)) {
            cout << "Result2: " << get<int>(result2) << endl;
        } else {
            cout << "Result2: " << get<string>(result2) << endl;
        }
    
        return 0;
    }
    ```
    
    这里使用了bool holds_alternative来判断返回的类型是什么。由于 Lambda 表达式有多个 `return` 语句且返回类型不同，我们必须显式指定返回类型为 `variant<int, string>`。
    
3. **还可以进一步简化为：[捕获列表] {函数体}**
    
    可以进一步忽略参数列表和返回类型，只保留捕获列表和函数体：
    
    `auto f = []{ return 1 + 2; };` 
    

## 2 捕获列表

当表达式需要使用外部变量的时候，用捕获列表来传参数，如：

```cpp
void test3()
{
    int c = 12;
    int d = 30;
    auto Add = [c, d](int a, int b)->int { //捕获列表加入使用的外部变量c，否则无法通过编译
        cout << "d = " << d  << endl;      //这里的c和d都是捕获的值，不是引用，假如c = a会报错
        return c;
    };
    d = 20;
    std::cout << Add(1, 2) << std::endl;
}
```

这里的[c, d]内有多种符号表示：

- 如果捕获列表为`[&]`，则表示所有的外部变量都按引用传递给lambda使用；
- 如果捕获列表为`[=]`，则表示所有的外部变量都按值传递给lambda使用；**这里的c和d就是值，对于按值传递的捕获列表，会立即将当前取到的值拷贝一份作为常数，然后将该常数作为参数传递。**
- `[this]` ，捕获当前类中的this指针，让lambda表达式拥有和当前类成员函数同样的访问权限，如果已经使用了 `&`或者 `=`, 默认添加此选项。

```cpp
void test4()
{
    int c = 12;
    int d = 30;
    auto Add = [&c, &d](int a, int b)->int { //捕获列表加入使用的外部变量c，否则无法通过编译
        c = a; // 编译对的
        cout << "d = " << d  << endl;
        return c;
    };
    d = 20;
    std::cout << Add(1, 2) << std::endl;
}
```

## 3 lamda表达式的本质

如果希望去修改按值捕获的外部变量，那么应该如何处理呢？这就需要使用mutable选项，**mutable修改是lambda表达式就算没有参数也要写明参数列表，并且可以去掉按值捕获的外部变量的只读（const）属性。**
```cpp
int a = 0;
auto f1 = [=] {return a++; };              // error, 按值捕获外部变量, a是只读的
auto f2 = [=]()mutable {return a++; };     // ok 
```