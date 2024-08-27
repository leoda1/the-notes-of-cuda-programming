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
/*erfen.cpp
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
}*/
/*高精度加法
#include <iostream>
#include <vector>
using namespace std;

const int N = 1e6 + 10;
//C = A + B
vector<int> add(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size() || i < B.size(); i++)
    {
        if (i < A.size()) t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back( t % 10);
        t /= 10;
    }
    if (t) C.push_back(t);
    return C;
}

int main()
{
    string a, b;
    vector<int> A, B;
    cin >> a >> b; // a = "123456" 
    //push_back就是把元素放到vector的容器里， -'0'是为了把字符转化为数字
    for (int i = a.size() - 1; i >= 0; i--) A.push_back(a[i] - '0');// A = [6, 5, 4, 3, 2, 1]
    for (int i = b.size() - 1; i >= 0; i--) B.push_back(b[i] - '0');

    auto C = add(A, B);
    for (int i = C.size() - 1; i >= 0; i--) cout << C[i];
    cout << endl;
    return 0;

}*/
/*高精度减法
#include <iostream>
#include <vector>
using namespace std;
const int N = 1e5 + 10;
//判断AB谁大
bool cmp(vector<int> &A, vector<int> &B)
{
    if (A.size() != B.size()) return A.size() > B.size();
    for (int i = A.size() - 1; i >= 0; i-- )
        if(A[i] != B[i]) return A[i] > B[i];
    return true;
}
// C = A - B
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i++)
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i]; 
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}

int main(){
    string a, b;
    cin >> a >> b;
    vector<int> A, B;
    for ( int i = a.size() - 1; i >= 0; i--) A.push_back(a[i] - '0');
    for ( int i = b.size() - 1; i >= 0; i--) B.push_back(b[i] - '0');
    if (cmp(A, B)){
        auto C = sub(A, B);
        for ( int i = C.size() - 1; i >= 0; i--) cout << C[i];
    }
    else{
        auto C = sub(B, A);
        cout << "-";
        for ( int i = C.size() - 1; i >= 0; i--) cout << C[i];
    }
    cout << endl;
    return 0;

}*/



#include <iostream>
using namespace std;
const int N = 100000;
int sum[N] = {0};

int main()
{
    int n, m, x;
    int a[N];
    cin >> n >> m;
    for (int i = 1; i <= n; i++){
        cin >> x;
        sum[i] = x + sum[i - 1];
    }
    while(m--){
        int l, r;
        cin >> l >> r;
        cout << sum[r] - sum[l - 1] << endl;
    }
    return 0;
}