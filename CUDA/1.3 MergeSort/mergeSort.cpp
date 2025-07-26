#include <iostream>
using namespace std;
const int N = 1000000;

int n;
int q[N], t[N];

//归并排序cpu测试
void mergeSort(int arr[], int l, int r)
{
    if(l >= r) return;
    int mid = (l + r) >> 1;
    mergeSort(arr, l, mid);
    mergeSort(arr, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while(i <= mid && j <= r){
        if(arr[i] <= arr[j]) t[k++] = arr[i++];
        else t[k++] = arr[j++];
    }
    while(i <= mid) t[ k++ ] = arr[ i++ ];
    while(j <= r) t[ k++ ] = arr[ j++ ];
    for (i = l, j = 0; i <= r; i++, j++) arr[i] = t[j];
}

int main() {
    cin >> n;
    for (int i = 0; i < n; i++) cin >> q[i];
    mergeSort(q, 0, n - 1);
    for (int i = 0; i < n; i++) cout << q[i] << " ";
    return 0;
}