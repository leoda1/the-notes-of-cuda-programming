#include <iostream>
#include <vector>

template<uint sortDir>
void insertionSort(std::vector<int>& data){
    for(size_t i = 1; i < data.size(); i++){
        int key = data[i];
        int j = i - 1;   // index of the last element in the sorted subarray
        while(j >= 0 && ((sortDir == 1 && data[j] > key) || (sortDir == 0 && data[j] < key))){
            data[j+1] = data[j];
            j--;
        }
        data[j+1] = key;
    }
}

int main(){
    std::vector<int> data = {5, 3, 8, 6, 2, 7, 1, 4};
    //asc
    insertionSort<1>(data);
    std::cout << "Ascending order :";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << " " << data[i];
    }
    std::cout << std::endl;
    //desc
    insertionSort<0>(data);
    std::cout << "Descending order:";
    for(size_t i = 0; i < data.size(); i++){
        std::cout << " " << data[i];
    }
    std::cout << std::endl;

    return 0;
}