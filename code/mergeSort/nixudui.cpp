#include <iostream>

using namespace std;
typedef long long LL;
const int N = 100000;
int n, q[N], t[N];

LL nixudui(int l, int r)
{
    if (l >= r) return 0;
    int mid = (l + r) >> 1;
    LL res = nixudui(l, mid) + nixudui(mid+1, r);
    //merge sort开始
    int k = 0, i = l, j = mid + 1;
    while ( i <= mid && j <= r)
        if (q[i] <= q[j]) t[k++] = q[i++];
        else
        {
            t[k++] = q[j++];
            res += mid - i + 1;
        }
    while (i <= mid) t[k++] = q[i++];
    while (j <= r) t[k++] = q[j++];
    for (int i = l; i <= r; i++) q[i] = t[i-l];
    //merge sort结束
    return res;
}

int main()
{
    cin >> n;
    for (int i = 0; i < n; i++) cin >> q[i];
    cout << nixudui(0, n-1) << endl;
    for (int i = 0; i < n; i++) cout << q[i];
    return 0;
}