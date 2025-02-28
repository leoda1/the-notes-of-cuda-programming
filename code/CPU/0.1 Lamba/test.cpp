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