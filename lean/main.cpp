#include <iostream>
#include <algorithm>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low;
    
    for (int j = low; j < high; j ++) {
        if (arr[j] < pivot) {
            swap(arr[i++], arr[j]);
        }
    }
    swap(arr[i], arr[high]);
    return i;
}

void quicksort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1,high);
    }
}

int main() {
    vector<int> arr = {10, 7, 8, 9, 1, 1, 5};
    int n = arr.size();

    quicksort(arr, 0, n - 1);
    cout << "Sorted array: ";
    for (auto & i : arr) {
       cout << i << " ";
    }
    return 0;
}