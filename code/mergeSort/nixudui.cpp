/*nixudui.cpp
#include <iostream>

using namespace std;
typedef long long LL;
const int N = 100000;
int n, q[N], t[N];

LL nixudui(int l, int r)
{
    if (l >= r) return 0;
    int mid = (l + r) >> 1;
    LL res = nixudui(l, mid) + nixudui(mid + 1, r);
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
    for (int i = l, j = 0; i <= r; i++, j++) q[i] = t[j];
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
}*/

#include <iostream>
using namespace std;
const int N = 100010;
int n, m;
int q[N];

int main(){
    scanf("%d%d", &n, &m);
    for ( int i = 0; i < n; i++ ) cin >> q[i];
    while ( m-- ) {
        int x;
        cin >> x;

        int l = 0, r = n-1;
        while ( l < r ){
            int mid = l + r >> 1;
            if ( q[mid] >= x ) r = mid;
            else l = mid + 1;
        }
        if (q[l] != x) cout << "-1 -1" << endl;
        else{
            cout << l << ' ';
            int l = 0, r = n - 1;
            while ( l < r){
                int mid = l + r + 1 >> 1;
                if ( q[mid] <= x ) l = mid;
                else r = mid - 1; 
            }
            cout << l << endl;
        }
    }
    return 0;
}