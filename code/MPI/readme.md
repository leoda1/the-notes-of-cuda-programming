# MPI Tutorial

# 1. MPI基本概念

## 1.1 abstract

消息传递模型？它其实只是指程序通过在进程间传递消息（消息可以理解成带有一些信息和数据的一个数据结构）来完成某些任务。MPI在消息传递模型设计上的经典概念：

- *通讯器（communicator）：*一组能够互相发消息的进程。在这组进程中，每个进程会被分配一个序号，称作*秩*（rank），进程间显性地通过指定秩来进行通信。
- *点对点*（point-to-point）通信：不同进程间发送和接收操作。一个进程可以通过指定另一个进程的秩以及一个独一无二的消息*标签*（*tag*）来发送消息给另一个进程。接受者可以发送一个接收特定标签标记的消息的请求（或者也可以完全不管标签，接收任何消息），然后依次处理接收到的数据。类似这样的涉及一个发送者以及一个接受者的通信被称作*点对点*（point-to-point）通信。
- *集体性*（collective）通信：单个进程和所有其他进程通信，类似广播机制，MPI专门的接口来实现这个过程所有进程间的集体性通信。API：
    
    
    | 操作 | 描述 | 结果返回给 |
    | --- | --- | --- |
    | `MPI_Bcast` | 广播 | 所有进程 |
    | `MPI_Gather` | 收集数据 | 根进程 |
    | `MPI_Allgather` | 收集数据 | 所有进程 |
    | `MPI_Scatter` | 分发数据 | 每个进程 |
    | `MPI_Reduce` | 归约操作 | 根进程 |
    | `MPI_Allreduce` | 归约操作 | 所有进程 |
    | `MPI_Scan` | 前缀归约 | 每个进程 |
    | `MPI_Gatherv` | 收集不同长度的数据 | 根进程 |
    | `MPI_Scatterv` | 分发不同长度的数据 | 每个进程 |

# 2. 点对点通信

## 2.1 send & rece info

MPI的send和rece的方式：开始的时候，*A* 进程决定要发送消息给 *B* 进程。A进程就会把需要发送给B进程的所有数据打包好，放到一个缓存里面。所有data会被打包到一个大的信息里，因此缓存会被比作*信封*（就像我们把好多信纸打包到一个信封里面然后再寄去邮局）。数据打包进缓存后，通信设备（通常是网络）就需要负责把信息传递到正确的地方，这个正确的地方是根据rank确定的那个进程。

尽管数据已经被送达到 B 了，但是进程 B 依然需要确认它想要接收 A 的数据。一旦确认了这点，数据就被传输成功了。进程A回接受到数据传递成功的消息，然后才去干其他事情。

当A需要传递不同的消息给B，为了让B能够方便区分不同的消息，运行发送者和接受者额外地指定一些信息 ID (正式名称是*标签*, *tags*)。当 B 只要求接收某种特定标签的信息的时候，其他的不是这个标签的信息会先被缓存起来，等到 B 需要的时候才会给 B。

`mpi_proto.h`中收发的定义：

```cpp
MPI_Send(
		const void *buf,       // 数据缓存（要发送的数据存储的内存地址）
    int count,             // 数量（MPI_Send 会精确地发送 count 指定的数量个元素）
    MPI_Datatype datatype, // 类型（发送的数据类型，例如 MPI_INT、MPI_FLOAT 等）
    int destination,       // 目标进程的 rank（表示要发送给哪个进程）
    int tag,               // 标签（用于标识消息的标签，发送和接收的标签需一致）
    MPI_Comm communicator) // 通信器（指定通信域，例如 MPI_COMM_WORLD）

MPI_Recv(
    void* data,            // 数据缓存（接收到的数据存储的内存地址）
    int count,             // 数量（MPI_Recv 期望接收 count 指定数量的数据）
    MPI_Datatype datatype, // 类型（接收的数据类型，例如 MPI_INT、MPI_FLOAT 等）
    int source,            // 源进程的 rank（表示从哪个进程接收数据，MPI_ANY_SOURCE 表示任意进程）
    int tag,               // 标签（用于标识消息的标签，MPI_ANY_TAG 表示任意标签）
    MPI_Comm communicator, // 通信器（指定通信域，例如 MPI_COMM_WORLD）
    MPI_Status* status)    // 状态（用于存储接收操作的返回状态，包括来源、标签、数据长度等信息）

```

## **2.2 基础 MPI 数据结构**

MPI 数据结构以及它们在 C 语言里对应的结构如下：

| MPI datatype | C equivalent |
| --- | --- |
| MPI_SHORT | short int |
| MPI_INT | int |
| MPI_LONG | long int |
| MPI_LONG_LONG | long long int |
| MPI_UNSIGNED_CHAR | unsigned char |
| MPI_UNSIGNED_SHORT | unsigned short int |
| MPI_UNSIGNED | unsigned int |
| MPI_UNSIGNED_LONG | unsigned long int |
| MPI_UNSIGNED_LONG_LONG | unsigned long long int |
| MPI_FLOAT | float |
| MPI_DOUBLE | double |
| MPI_LONG_DOUBLE | long double |
| MPI_BYTE | char |

## 2.3 send和rece的两个范例：

exp1的代码：

```cpp
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int number;
    if (world_rank == 0) {
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1){
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n", number);
    }
    MPI_Finalize();
    return 0;
}
/******************************************************************
base) joker@joker-2 4.1 Send & Recv % mpirun -np 2 ./send_recv
Process 1 received number -1 from process 0
(base) joker@joker-2 4.1 Send & Recv % mpirun -np 1 ./send_recv
World size must be greater than 1 for ./send_recv
Abort(1) on node 0 (rank 0 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 1) - process 0
*******************************************************************/
```

代码当中的`MPI_Comm_rank`用于获取当前进程在给定通信子（communicator）中的 **rank**（进程编号）。`rank` 是进程在 `comm`（通信器）中的编号，编号从 `0` 开始。`MPI_COMM_WORLD` 是默认通信器，表示所有进程都属于这个通信器。`MPI_Comm_size`是用于获取在给定通信器 `comm` 中的总进程数。`MPI_Abort`是断言，强制终止通信器的所有进程，第二参数是非0值的话表示异常退出。`stderr` 是 C/C++ 中的标准错误流（standard error），在 `fprintf` 中，`stderr` 是立即输出的，不受缓冲区控制。`MPI_STATUS_IGNORE`表示当前的`MPI_Recv` 不关心接收的状态信息。

代码逻辑：总线程数<2的情况直接`abort`，然后正常情况下的话直接初始化一个number为-1从线程0发送到线程1。

exp2的代码:

```cpp
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv) {
    const int PING_PONG_LIMIT = 10;

    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        fprintf(stderr, "World size must be 2 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int ping_pong_count = 0;
    int partner_rank = (world_rank + 1) % 2;
    while (ping_pong_count < PING_PONG_LIMIT) {
        if (world_rank == ping_pong_count % 2) {
            // Increment the ping pong count before you send it
            ping_pong_count++;
            MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
            printf("%d sent and incremented ping_pong_count %d to %d\n",
                world_rank, ping_pong_count,
                partner_rank);
        } else {
            MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%d received ping_pong_count %d from %d\n",
                world_rank, ping_pong_count, partner_rank);
        }
    }
}
/******************************************************************
1 received ping_pong_count 1 from 0
0 sent and incremented ping_pong_count 1 to 1
1 sent and incremented ping_pong_count 2 to 0
0 received ping_pong_count 2 from 1
0 sent and incremented ping_pong_count 3 to 1
0 received ping_pong_count 4 from 1
0 sent and incremented ping_pong_count 5 to 1
0 received ping_pong_count 6 from 1
0 sent and incremented ping_pong_count 7 to 1
0 received ping_pong_count 8 from 1
0 sent and incremented ping_pong_count 9 to 1
0 received ping_pong_count 10 from 1
1 received ping_pong_count 3 from 0
1 sent and incremented ping_pong_count 4 to 0
1 received ping_pong_count 5 from 0
1 sent and incremented ping_pong_count 6 to 0
1 received ping_pong_count 7 from 0
1 sent and incremented ping_pong_count 8 to 0
1 received ping_pong_count 9 from 0
1 sent and incremented ping_pong_count 10 to 0
*******************************************************************/
```

进程0和进程1在轮流发送和接收 ping_pong_count。

## **2.4 MPI Status & Probe动态的接收**

`MPI_Recv`将`MPI_Status`结构体的地址作为参数，可以使用`MPI_STATUS_IGNORE` 忽略。如果我们将 `MPI_Status` 结构体传递给 `MPI_Recv` 函数，则操作完成后将在该结构体中填充有关接收操作的其他信息。 三个主要的信息包括：

- 发送端的rank，发送端的rank存储在结构体`MPI_SOURCE`元素中，如果声明一个`MPI_Status state`对象，则可以通过state.MPI_SOURCE访问rank。
    
    ```cpp
    typedef struct MPI_Status {
        int count_lo;                  // 低位的计数值，表示接收到的数据量的低32位（可能与 count_hi_and_cancelled 组合成完整的 64 位计数）
        int count_hi_and_cancelled;    // 高位的计数值（如果存在高32位），同时包含一个“取消标志”位
        int MPI_SOURCE;                // 消息的源进程的 rank（表示接收消息是从哪个进程来的）
        int MPI_TAG;                   // 消息的标签（与发送时指定的标签对应，用于标识消息的类型）
        int MPI_ERROR;                 // 错误码（用于存储接收操作的返回状态，MPI_SUCCESS 表示成功）
    } MPI_Status;
    ```
    
- 消息的标签，同上访问方式，访问`MPI_TAG`。
- 消息的长度，它在在结构体中没有预定义的元素。我们必须使用 `MPI_Get_count` 找出消息的长度。
    
    ```cpp
    MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)
    ```
    
    在`MPI_Get_count`中需要传递 `MPI_Status` 结构体，消息的 `datatype`（数据类型），并返回 `count`。 变量 `count` 是已接收的 `datatype` 元素的数目。
    

### 2.4.1 MPI_Status结构体查询的范例

```cpp
// mpicc mpi_status.cc -o mpi_status
// mpirun -np 2 ./mpi_status
#include <mpi.h>
#include <iostream>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
  
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
      fprintf(stderr, "Must use two processes for this example\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    const int MAX_NUMBERS = 100;
    int numbers[MAX_NUMBERS];
    int number_amount;
    if (world_rank == 0) {
      // Pick a random amount of integers to send to process one
      srand(time(NULL));
      number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;
      // Send the amount of integers to process one
      MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
      printf("0 sent %d numbers to 1\n", number_amount);
    } else if (world_rank == 1) {
      MPI_Status status;
      // Receive at most MAX_NUMBERS from process zero
      MPI_Recv(numbers, MAX_NUMBERS, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      // After receiving the message, check the status to determine how many
      // numbers were actually received
      MPI_Get_count(&status, MPI_INT, &number_amount);
      // Print off the amount of numbers, and also print additional information
      // in the status object
      printf("1 received %d numbers from 0. Message source = %d, tag = %d\n",
             number_amount, status.MPI_SOURCE, status.MPI_TAG);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
/******************************************************************
0 sent 91 numbers to 1
0 sent 91 numbers to 1
*******************************************************************/
```

### 2.4.2 use MPI_Probe找出消息大小

在库文件中的定义如下，可以看到与`MPI_Recv`很类似。可以使用 `MPI_Probe` 在实际接收消息之前查询消息大小。除了不接收消息之外，`MPI_Probe`会阻塞具有匹配标签和发送端的消息。消息可用时，会填充`Status`。然后，用户可以使用 `MPI_Recv` 接收实际的消息。 

```cpp
MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status)
```

```cpp
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
    if (world_size != 2) {
        fprintf(stderr, "Error: This program requires exactly 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the current process

    int number_of_amount = 0;
    if (world_rank == 0) {
        int MAX_NUMBERS = 100;
        int numbers[MAX_NUMBERS];
        srand(time(NULL));
        number_of_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;
        MPI_Send(numbers, number_of_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d sent %d numbers to process %d.\n", world_rank, number_of_amount, 1);
    } else if (world_rank == 1) {
        MPI_Status status;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &number_of_amount);
        int* number_buffer = (int*)malloc(sizeof(int) * number_of_amount);
        MPI_Recv(number_buffer, number_of_amount, MPI_INT, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
            printf("1 dynamically received %d numbers from 0.\n",
                number_of_amount);
        free(number_buffer);
    }
    MPI_Finalize();
}
/******************************************************************
Process 0 sent 29 numbers to process 1.
1 dynamically received 29 numbers from 0.
*******************************************************************/
```

`MPI_Probe` 构成了许多动态 MPI 应用程序的基础。 例如，控制端/执行子程序在交换变量大小的消息时通常会大量使用 `MPI_Probe`。 作为练习，对 `MPI_Recv` 进行包装，将 `MPI_Probe` 用于您可能编写的任何动态应用程序。 它将使代码看起来更美好：-)

### 2.5 阻塞通信的发生和解决

MPI的p2p中包括两种模式：

1. 阻塞通信（Blocking）
    
    `MPI_Send` 和 `MPI_Recv` 在返回前，必须满足以下条件之一：`MPI_Send` 完成消息的发送（或在缓冲区中完成存储）和`MPI_Recv` 完成消息的接收（数据已经被复制到接收缓冲区中）。此时，直到完成发送和接收，这两个函数会让进程在此处“停住”。
    
2. 非阻塞通信（Non-blocking）
    
    发送和接收操作被立即返回，通信操作在后台继续进行。需要通过 `MPI_Test` 或 `MPI_Wait` 等函数来判断通信是否完成。
    

### 2.6 点对点通信应用程序案例 - 随机步行（Random Walk）

问题描述：给定左右边界Min，Max和游走器Walker。游走器 *W* 向右以任意长度的 *S* 随机移动。 如果该过程越过边界，它就会绕回。如何并行化随机游走问题？

![1](http://stxg6c3mb.hd-bkt.clouddn.com/image.png)
首先在各个进程之间划分域。 并将域的大小定义为Max - Min + 1，（因为游走器包含 *Max* 和 *Min*）。 假设游走器只能采取整数大小的步长，我们可以轻松地将域在每个进程中划分为大小近乎相等的块。 例如，如果 *Min* 为 0，*Max* 为 20，并且我们有四个进程，则将像这样拆分域。

![2](http://stxg6c3mb.hd-bkt.clouddn.com/截屏2025-03-19_13.55.38.png)

012每个域有五个单元，最后一个域有6个单元。例如，如果游走器在进程 0（使用先前的分解域）上进行了移动总数为 6 的游走，则游走器的执行将如下所示：

1. 游走器的步行长度开始增加。但是，当它的值达到 4 时，它已到达进程 0 的边界。因此，进程 0 必须与进程 1 交流游走器的信息。
2. 进程 1 接收游走器，并继续移动，直到达到移动总数 6。然后，游走器可以继续进行新的随机移动。

*W* 仅需从进程 0 到进程 1 进行一次通信。 

```cpp
#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <time.h>

using namespace std;

void decompose_domain (int domain_size, int world_rank, int world_size,
                       int * subdomain_start, int * subdomain_size) {
    /* 将域分割成偶数个块，函数返回子域开始和子域大小 */
    if (world_size > domain_size) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *subdomain_start = domain_size / world_size * world_rank;
    *subdomain_size = domain_size / world_size;
    if (world_rank == world_size - 1) {
        *subdomain_size += domain_size % world_size;
    }
}

typedef struct {
    int location;
    int num_step_left_in_walk;
} Walker;

void initialize_walker (int num_walker_per_proc, int max_walk_size, int subdomain_start,
                        vector<Walker> *incoming_walkers) {
    Walker walker;
    for (int i = 0; i < num_walker_per_proc; i++) {
        // Initialize walkers in the middle of the subdomain
        walker.location = subdomain_start;
        walker.num_step_left_in_walk = (rand() / (float)RAND_MAX) * max_walk_size;
        incoming_walkers->push_back(walker);
    }
}

void walk(Walker* walker, int subdomain_start, int subdomain_size,
    int domain_size, vector<Walker>* outgoing_walkers) {
    while (walker->num_step_left_in_walk > 0) {
        if (walker->location >= subdomain_start + subdomain_size) {
        // Take care of the case when the walker is at the end
        // of the domain by wrapping it around to the beginning
            if (walker->location == domain_size) {
                walker->location = 0;
            }
            outgoing_walkers->push_back(*walker);
            break;
        } else {
            walker->num_step_left_in_walk--;
            walker->location++;
        }
    }
}

void send_outgoing_walkers(vector<Walker>* outgoing_walkers,
                     int world_rank, int world_size) {
    // Send the data as an array of MPI_BYTEs to the next process.
    // The last process sends to process zero.
    MPI_Send((void*)outgoing_walkers->data(),
        outgoing_walkers->size() * sizeof(Walker), MPI_BYTE,
        (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
    // Clear the outgoing walkers list
    outgoing_walkers->clear();
}

void receive_incoming_walkers(vector<Walker>* incoming_walkers,
                        int world_rank, int world_size) {
    // Probe for new incoming walkers
    MPI_Status status;
    // Receive from the process before you. If you are process zero,
    // receive from the last process
    int incoming_rank = (world_rank == 0) ? world_size - 1 : world_rank - 1;
    MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status);
    // Resize your incoming walker buffer based on how much data is
    // being received
    int incoming_walkers_size;
    MPI_Get_count(&status, MPI_BYTE, &incoming_walkers_size);
    incoming_walkers->resize(incoming_walkers_size / sizeof(Walker));
    MPI_Recv((void*)incoming_walkers->data(), incoming_walkers_size,
        MPI_BYTE, incoming_rank, 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
}

int main(int argc, char** argv) {
    int domain_size;
    int max_walk_size;
    int num_walkers_per_proc;

    if (argc < 4) {
        cerr << "Usage: random_walk domain_size max_walk_size "
             << "num_walkers_per_proc" << endl;
        exit(1);
    }

    domain_size = atoi(argv[1]);
    max_walk_size = atoi(argv[2]);
    num_walkers_per_proc = atoi(argv[3]);

    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(time(NULL) * world_rank);

    int subdomain_start, subdomain_size;
    vector<Walker> incoming_walkers, outgoing_walkers;

    // Find your part of the domain
    decompose_domain(domain_size, world_rank, world_size,
                     &subdomain_start, &subdomain_size);

    // Initialize walkers in your subdomain
    initialize_walker(num_walkers_per_proc, max_walk_size, subdomain_start,
                      &incoming_walkers);

    cout << "Process " << world_rank << " initiated " << num_walkers_per_proc
         << " walkers in subdomain " << subdomain_start << " - "
         << subdomain_start + subdomain_size - 1 << endl;

    // Determine the maximum amount of sends and receives needed to
    // complete all walkers
    int maximum_sends_recvs = max_walk_size / (domain_size / world_size) + 1;
    for (int m = 0; m < maximum_sends_recvs; m++) {
        // Process all incoming walkers
        for (int i = 0; i < incoming_walkers.size(); i++) {
            walk(&incoming_walkers[i], subdomain_start, subdomain_size,
                 domain_size, &outgoing_walkers);
        }

        cout << "Process " << world_rank << " sending " << outgoing_walkers.size()
             << " outgoing walkers to process " << (world_rank + 1) % world_size
             << endl;

        if (world_rank % 2 == 0) {
            // Send all outgoing walkers to the next process.
            send_outgoing_walkers(&outgoing_walkers, world_rank,
                                  world_size);
            // Receive all the new incoming walkers
            receive_incoming_walkers(&incoming_walkers, world_rank,
                                     world_size);
        } else {
            // Receive all the new incoming walkers
            receive_incoming_walkers(&incoming_walkers, world_rank,
                                     world_size);
            // Send all outgoing walkers to the next process.
            send_outgoing_walkers(&outgoing_walkers, world_rank,
                                  world_size);
        }

        cout << "Process " << world_rank << " received " << incoming_walkers.size()
             << " incoming walkers" << endl;
    }

    cout << "Process " << world_rank << " done" << endl;

    MPI_Finalize();
    return 0;
}
// instruction
// mpic++ random_walk.cc -o random_walk
// mpirun -np 5 ./random_walk 100 500 20
```

![3](http://stxg6c3mb.hd-bkt.clouddn.com/IMG_2762.heic)

# 3. 集体通信

集体通信指的是一个涉及 communicator 里面所有进程的一个方法。关于集体通信需要记住的一点是它在进程间引入了同步点的概念。这意味着所有的进程在执行代码的时候必须首先*都*到达一个同步点才能继续执行后面的代码。MPI 有一个特殊的函数来做同步进程的这个操作：

```cpp
MPI_Barrier(MPI_Comm comm)
```

这个方法会构建一个屏障，任何进程都没法跨越屏障，直到所有的进程都到达屏障。这边有一个示意图。假设水平的轴代表的是程序的执行，小圆圈代表不同的进程。

![4](http://stxg6c3mb.hd-bkt.clouddn.com/image 1.png)

这里四个时间内不同进程的执行逻辑是：进程0在时间点 (T 1) 首先调用 `MPI_Barrier`。然后进程0就一直等在屏障之前，之后进程1和进程3在 (T 2) 时间点到达屏障。当进程2最终在时间点 (T 3) 到达屏障的时候，其他的进程就可以在 (T 4) 时间点再次开始运行。`MPI_Barrier` 在很多时候很有用。其中一个用途是用来同步一个程序，使得分布式代码中的某一部分可以被精确的计时。

> 注意：在 MPI 中，**所有的集体通信**（如 `MPI_Bcast`、`MPI_Gather`、`MPI_Scatter` 等）**都是同步的**，也就是说：必须让**所有相关进程**都参与到同一次集体通信中，否则有一个进程掉队，其他进程就会一直等待，导致死锁。
> 

## 3. 1 MPI_Bcast广播

*广播* (broadcast) 是标准的集体通信技术之一。一个广播发生的时候，一个进程会把同样一份数据传递给一个 communicator 里的所有其他进程。广播的主要用途之一是把用户输入传递给一个分布式程序，或者把一些配置参数传递给所有的进程。它的函数签名：

```cpp
MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
					int root, // 广播的根进程的rank值（进程号）
					MPI_Comm comm)
```

当根节点(在我们的例子是节点0)调用 `MPI_Bcast` 函数的时候，`buffer` 变量里的值会被发送到其他的节点上。当其他的节点调用 `MPI_Bcast` 的时候，`buffer` 变量会被赋值成从根节点接受到的数据。我们可以使用`MPI_Send`和 `MPI_Recv`来实现广播，代码很简单如下：

```cpp
void my_bcast(void* data, int count, MPI_Datatype datatype, int root,
              MPI_Comm communicator) {
  int world_rank;
  MPI_Comm_rank(communicator, &world_rank);
  int world_size;
  MPI_Comm_size(communicator, &world_size);

  if (world_rank == root) {
    // If we are the root process, send our data to everyone
    int i;
    for (i = 0; i < world_size; i++) {
      if (i != world_rank) {
        MPI_Send(data, count, datatype, i, 0, communicator);
      }
    }
  } else {
    // If we are a receiver process, receive the data from the root
    MPI_Recv(data, count, datatype, root, 0, communicator,
             MPI_STATUS_IGNORE);
  }
}
```

根节点把数据传递给所有其他的节点，其他的节点接收根节点的数据。但是这里的效率很低，因为每次并不是一下就完成了所有的进程发送和接收。只是使用了进程0的一次次的传递数据。这里有一些优化算法，如：基于树的沟通算法。

## 3.2 MPI_**Scatter,** MPI_**Gather, and** MPI_**Allgather**

### 3.2.1 MPI_Scatter

`MPI_Scatter`是一个类似`MPI_Bcast`的集体通信机制。它会会设计一个指定的根进程，根进程会将数据发送到 communicator 里面的所有进程，但是给每个进程发送的是*一个数组的一部分数据*。

![5](http://stxg6c3mb.hd-bkt.clouddn.com/image 2.png)

`MPI_Bcast` 在根进程上接收一个单独的数据元素，复制给其他进程。

`MPI_Scatter` 接收一个数组，并把元素按进程的秩分发出去。尽管根进程（进程0）拥有整个数组的所有元素，`MPI_Scatter` 还是会把正确的属于进程0的元素放到这个进程的接收缓存中。

```cpp
int MPI_Scatter(
		const void *sendbuf,   // 发送缓存（存储要发送的数据的起始地址）
    int sendcount,         // 发送数据的数量（每个进程接收的元素个数）
    MPI_Datatype sendtype, // 发送数据的类型（如 MPI_INT, MPI_FLOAT 等）
    void *recvbuf,         // 接收缓存（存储接收到的数据的起始地址）
    int recvcount,         // 接收数据的数量（每个进程接收的元素个数）
    MPI_Datatype recvtype, // 接收数据的类型（如 MPI_INT, MPI_FLOAT 等）
    int root,              // 根进程（发送数据的源进程）
    MPI_Comm comm)         // 通信器（指定通信域，如 MPI_COMM_WORLD）
```

### 3.2.2 MPI_**Gather**

顾名思义这里的`MPI_Gather`是和`MPI_Scatter` 相反的。它是从多个进程里面收集数据到一个进程上面，这个机制对很多平行算法很有用，比如并行的排序和搜索。如图：

![5](http://stxg6c3mb.hd-bkt.clouddn.com/image 3.png)

元素是根据接收到的进程的秩排序的。函数签名如下：

```cpp
int MPI_Gather(
    const void *sendbuf,   // 发送缓冲区（存储要发送的数据的起始地址）
    int sendcount,         // 发送数据的数量（每个进程发送的数据元素个数）
    MPI_Datatype sendtype, // 发送数据的类型（如 MPI_INT, MPI_FLOAT 等）
    void *recvbuf,         // 接收缓冲区（存储接收到的数据的起始地址，只有 root 进程需要设置）
    int recvcount,         // 接收数据的数量（从每个进程接收的数据元素个数）
    MPI_Datatype recvtype, // 接收数据的类型（如 MPI_INT, MPI_FLOAT 等）
    int root,              // 根进程（用于收集数据的目标进程）
    MPI_Comm comm          // 通信器（指定通信域，如 MPI_COMM_WORLD）
);
```

只有根进程需要一个有效的接收缓存。所有其他的调用进程可以传递`NULL`给`recvbuf`。另外，别忘记`*recvcount*`参数是从*每个进程*接收到的数据数量，而不是所有进程的数据总量之和。

> **范例：用`Scatter`和Gather计算平均数**
> 

首先生成一个随机数的数组，scatter给不同进程，每个进程的到相同多数量的随机数，每个进程计算各自的avg，然后最后求总的avg。

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    assert(rand_nums != NULL);
    int i;
    for (i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float)RAND_MAX);
    }
    return rand_nums;
}

// Computes the average of an array of numbers
float compute_avg(float *array, int num_elements) {
    float sum = 0.f;
    for (int i = 0; i < num_elements; i++) {
        sum += array[i];
    }
    return sum / num_elements;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: avg num_elements_per_proc\n");
        exit(1);
    }

    int num_elements_per_proc = atoi(argv[1]);
    // Seed the random number generator to get different results each time
    srand(time(NULL));

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Create a random array of elements on the root process. Its total
    // size will be the number of elements per process times the number
    // of processes
    float *rand_nums = NULL;
    if (world_rank == 0) {
        rand_nums = create_rand_nums(num_elements_per_proc * world_size);
    }

    // For each process, create a buffer that will hold a subset of the entire array
    float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_per_proc);
    assert(sub_rand_nums != NULL);

    // Scatter the random numbers from the root process to all processes in the MPI world
    MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums,
                num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute the average of your subset
    float sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);

    // Gather all partial averages down to the root process
    float *sub_avgs = NULL;
    if (world_rank == 0) {
        sub_avgs = (float *)malloc(sizeof(float) * world_size);
        assert(sub_avgs != NULL);
    }
    MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Now that we have all of the partial averages on the root, compute the
    // total average of all numbers. Since we are assuming each process computed
    // an average across an equal amount of elements, this computation will
    // produce the correct answer.
    if (world_rank == 0) {
        float avg = compute_avg(sub_avgs, world_size);
        printf("Avg of all elements is %f\n", avg);

        // Compute the average across the original data for comparison
        float original_data_avg = compute_avg(rand_nums, num_elements_per_proc * world_size);
        printf("Avg computed across original data is %f\n", original_data_avg);
    }

    // Clean up
    if (world_rank == 0) {
        free(rand_nums);
        free(sub_avgs);
    }
    free(sub_rand_nums);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
/******************************************************************
(base) joker@joker-2 4.2 Collective % mpic++ avg_scatter_gather.cc -o avg_scatter_gather    
(base) joker@joker-2 4.2 Collective % mpirun -np 4 ./avg_scatter_gather 2
Avg of all elements is 0.444133
Avg computed across original data is 0.444133
*******************************************************************/
```

### 3.2.3 MPI_**Allgather**

前面出现了一对多，多对一，一对一等的通信模式，那么`MPI_Allgather`就是多对多。准确来说是收集所有进程的数据然后发到所有进程上，不涉及根进程了，所以可以看到函数签名里面少了`int root`。

![6](http://stxg6c3mb.hd-bkt.clouddn.com/image 4.png)

```cpp
int MPI_Allgather(
    const void *sendbuf,   // 发送缓冲区（存储要发送的数据的起始地址）
    int sendcount,         // 发送数据的数量（每个进程发送的数据元素个数）
    MPI_Datatype sendtype, // 发送数据的类型（如 MPI_INT, MPI_FLOAT 等）
    void *recvbuf,         // 接收缓冲区（存储接收到的数据的起始地址）
    int recvcount,         // 每个进程接收的数据数量
    MPI_Datatype recvtype, // 接收数据的类型（如 MPI_INT, MPI_FLOAT 等）
    MPI_Comm comm          // 通信器（指定通信域，如 MPI_COMM_WORLD）
);
```

对上一个计算平均数的代码修改，将最后输出的每个线程都是全局平均值。代码如下：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    assert(rand_nums != NULL);
    int i;
    for (i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float)RAND_MAX);
    }
    return rand_nums;
}

// Computes the average of an array of numbers
float compute_avg(float *array, int num_elements) {
    float sum = 0.f;
    for (int i = 0; i < num_elements; i++) {
        sum += array[i];
    }
    return sum / num_elements;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: avg num_elements_per_proc\n");
        exit(1);
    }

    int num_elements_per_proc = atoi(argv[1]);
    // Seed the random number generator to get different results each time
    srand(time(NULL));

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Create a random array of elements on the root process. Its total
    // size will be the number of elements per process times the number
    // of processes
    float *rand_nums = NULL;
    if (world_rank == 0) {
        rand_nums = create_rand_nums(num_elements_per_proc * world_size);
    }

    // For each process, create a buffer that will hold a subset of the entire array
    float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_per_proc);
    assert(sub_rand_nums != NULL);

    // Scatter the random numbers from the root process to all processes in the MPI world
    MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums,
                num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute the average of your subset
    float sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);

    // Gather all partial averages down to the root process
    float *sub_avgs = (float*)malloc(sizeof(float) * num_elements_per_proc);
    assert(sub_avgs != NULL);
    MPI_Allgather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, MPI_COMM_WORLD);

    // Now that we have all of the partial averages on the root, compute the
    // total average of all numbers. Since we are assuming each process computed
    // an average across an equal amount of elements, this computation will
    // produce the correct answer.
    float avg = compute_avg(sub_avgs, world_size);
    printf("Avg of all elements from proc is %f\n", avg);

    // Compute the average across the original data for comparison
    // float original_data_avg = compute_avg(rand_nums, num_elements_per_proc * world_size);
    // printf("Avg computed across original data is %f\n", original_data_avg);
    

    // Clean up
    if (world_rank == 0) {
        free(rand_nums);
        free(sub_avgs);
    }
    free(sub_rand_nums);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
/******************************************************************
(base) joker@joker-2 4.2 Collective % mpic++ avg_allgather.cc -o avg_allgather
(base) joker@joker-2 4.2 Collective % mpirun -np 4 ./avg_allgather 3          
Avg of all elements from proc is 0.579840
Avg of all elements from proc is 0.579840
Avg of all elements from proc is 0.579840
Avg of all elements from proc is 0.579840
*******************************************************************/
```

# **4. 高级集体通讯**

## 4.1 MPI_Reduce

前言：reduce就是规约，例如数组[1,2,3]的求和规约就是6，求平均规约就是2。

`MPI_Reduce` 在每个进程上获取一个输入元素数组，并将输出元素数组返回给根进程。原型是：

```cpp
int MPI_Reduce(
    const void *sendbuf,   // 发送缓冲区（存储要发送的数据的起始地址）
    void *recvbuf,         // 接收缓冲区（存储归约结果的起始地址，只有 root 进程需要设置）
    int count,             // 发送和接收的数据数量
    MPI_Datatype datatype, // 数据类型（如 MPI_INT、MPI_FLOAT 等）
    MPI_Op op,             // 归约操作（如 MPI_SUM、MPI_MAX、MPI_MIN 等）
    int root,              // 根进程（存储归约结果的目标进程）
    MPI_Comm comm          // 通信器（指定通信域，如 MPI_COMM_WORLD）
);
```

`op` 参数如下：

- `MPI_MAX` - 返回最大元素。
- `MPI_MIN` - 返回最小元素。
- `MPI_SUM` - 对元素求和。
- `MPI_PROD` - 将所有元素相乘。
- `MPI_LAND` - 对元素执行逻辑*与*运算。
- `MPI_LOR` - 对元素执行逻辑*或*运算。

- `MPI_BAND` - 对元素的各个位按位*与*执行。
- `MPI_BOR` - 对元素的位执行按位*或*运算。
- `MPI_MAXLOC` - 返回最大值和所在的进程的秩。
- `MPI_MINLOC` - 返回最小值和所在的进程的秩。

![7](http://stxg6c3mb.hd-bkt.clouddn.com/image 5.png)

范例代码如下：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>

float* create_rand_nums (int num_elements) {
    float* rand_nums = (float*)malloc(num_elements * sizeof(float));
    assert(rand_nums != NULL);
    for (int i = 0; i < num_elements; i ++) {
        rand_nums[i] = ((float) rand() / RAND_MAX);
    }
    return rand_nums;
}

int main (int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: avg num_elements_per_proc\n");
        exit(1);
    }

    int num_ele_per_proc = atoi(argv[1]);
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(time(NULL) * world_rank);
    float *rand_nums = NULL;
    rand_nums = create_rand_nums(num_ele_per_proc);

    float local_sum = 0;
    for (int i = 0; i < num_ele_per_proc; i ++) {
        local_sum += rand_nums[i];
    }
    printf("Local sum for process %d - %f, avg = %f\n",
        world_rank, local_sum, local_sum / num_ele_per_proc);
    
    float global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Total sum = %f, avg = %f\n", global_sum, global_sum / (world_size * num_ele_per_proc));
    }
    
    // Clean up
    free(rand_nums);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
/******************************************************************
(base) joker@joker-2 4.3 Advanced collective % mpic++ MPI_Reduce_exp1.cc -o MPI_Reduce_exp1
(base) joker@joker-2 4.3 Advanced collective % mpirun -np 4 ./MPI_Reduce_exp1 100
Local sum for process 1 - 48.185650, avg = 0.481856
Local sum for process 2 - 52.371292, avg = 0.523713
Local sum for process 3 - 52.872005, avg = 0.528720
Local sum for process 0 - 51.385098, avg = 0.513851
Total sum = 204.814056, avg = 0.512035
*******************************************************************/
```

## 4.2 **MPI_Allreduce**

刚刚的范例可以看到最后规约的结果是store在rank0上的，所以`ALLreduce`就是的出现就是为了让所有进程访问规约的结果。函数原型是：

```cpp
int MPI_Allreduce(
    const void *sendbuf,   // 发送缓冲区（存储要发送的数据的起始地址）
    void *recvbuf,         // 接收缓冲区（存储归约结果的起始地址，每个进程都会接收相同的结果）
    int count,             // 发送和接收的数据数量
    MPI_Datatype datatype, // 数据类型（如 MPI_INT、MPI_FLOAT 等）
    MPI_Op op,             // 归约操作（如 MPI_SUM、MPI_MAX、MPI_MIN 等）
    MPI_Comm comm          // 通信器（指定通信域，如 MPI_COMM_WORLD）
);
```

它不需要根进程 ID（因为结果分配给所有进程）。 下图介绍了 `MPI_Allreduce` 的通信模式：

![7](http://stxg6c3mb.hd-bkt.clouddn.com/image 6.png)

相当于先执行了`MPI_Reduce`之后，再执行`MPI_Bcast`。

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    assert(rand_nums != NULL);
    int i;
    for (i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float)RAND_MAX);
    }
    return rand_nums;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: avg num_elements_per_proc\n");
        exit(1);
    }

    int num_elements_per_proc = atoi(argv[1]);

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Create a random array of elements on all processes.
    srand(time(NULL) * world_rank); // Seed the random number generator of processes uniquely
    float *rand_nums = NULL;
    rand_nums = create_rand_nums(num_elements_per_proc);

    // Sum the numbers locally
    float local_sum = 0;
    int i;
    for (i = 0; i < num_elements_per_proc; i++) {
        local_sum += rand_nums[i];
    }

    // Reduce all of the local sums into the global sum in order to calculate the mean
    float global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float mean = global_sum / (num_elements_per_proc * world_size);

    // Compute the local sum of the squared differences from the mean
    float local_sq_diff = 0;
    for (i = 0; i < num_elements_per_proc; i++) {
        local_sq_diff += (rand_nums[i] - mean) * (rand_nums[i] - mean);
    }

    // Reduce the global sum of the squared differences to the root process
    // and print off the answer
    float global_sq_diff;
    MPI_Reduce(&local_sq_diff, &global_sq_diff, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // The standard deviation is the square root of the mean of the squared differences
    if (world_rank == 0) {
        float stddev = sqrt(global_sq_diff / (num_elements_per_proc * world_size));
        printf("Mean = %f, Standard deviation = %f\n", mean, stddev);
    }

    // Clean up
    free(rand_nums);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
/******************************************************************
(base) joker@joker-2 4.3 Advanced collective % mpic++ MPI_AllReduce.cc -o MPI_AllReduce    
(base) joker@joker-2 4.3 Advanced collective % mpirun -np 4 ./MPI_AllReduce 100        
Mean = 0.507307, Standard deviation = 0.289248
*******************************************************************/

```

# 5 group和communicator

当程序规模开始变大时，我们可能只想一次与几个进程进行对话。所以如何创建新的通讯器，以便一次与原始线程组的子集进行沟通？

### 5.1 创建新的通讯器

想对网格中进程的子集执行计算。 例如，每一行中的所有进程都可能希望对一个值求和。 这将是第一个也是最常见的用于创建新的通讯器的函数：

```cpp
int MPI_Comm_split(
    MPI_Comm comm,   // 输入的通信器（用于划分的基础通信域）
    int color,       // 划分标识（相同 color 的进程会被分到同一个通信器中）
    int key,         // 排序标识（在新通信器中根据 key 值进行排序）
    MPI_Comm* newcomm // 输出的新通信器（存储新的通信器）
);
```

`MPI_Comm_split` 通过基于输入值 `color` 和 `key` 将通讯器“拆分”为一组子通讯器来创建新的通讯器。 在这里需要注意的是，原始的通讯器并没有消失，但是在每个进程中都会创建一个新的通讯器。 第一个参数 `comm` 是通讯器，它将用作新通讯器的基础。 这可能是 `MPI_COMM_WORLD`，但也可能是其他任何通讯器。 第二个参数 `color` 确定每个进程将属于哪个新的通讯器。 为 `color` 传递相同值的所有进程都分配给同一通讯器。 如果 `color` 为 `MPI_UNDEFINED`，则该进程将不包含在任何新的通讯器中。 第三个参数 `key` 确定每个新通讯器中的`rank`顺序。 传递 `key` 最小值的进程将为 0，下一个最小值将为 1，依此类推。 如果存在平局，则在原始通讯器中秩较低的进程将是第一位。 最后一个参数 `newcomm` 是 MPI 如何将新的通讯器返回给用户。

## 5.2 使用多个通讯器

示例中，我们尝试将单个全局通讯器拆分为一组较小的通讯器。 在此示例中，我们将想象我们已经在逻辑上将原始通讯器布局为共 16 个进程的 4x4 网格，并且希望按行划分网格。 为此，每一行将获得自己的颜色（参数 `color`）。 在下图中，您可以看到左图具有相同颜色的每组进程如何最终变成右图的自己的通讯器。

![8](http://stxg6c3mb.hd-bkt.clouddn.com/image 7.png)

```cpp
// 代码实现
int world_rank, world_size; // 原始通讯器的rank 和 size
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

int color = world_rank / 4; // 根据行确定颜色
MPI_Comm row_comm;
MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &row_comm);//这里的key是world_rank

int row_rank, row_size;
MPI_Comm_rank(row_comm, &row_rank);
MPI_Comm_size(row_comm, &row_size);

printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d\n", world_rank, world_size, row_rank, row_size);
MPI_Comm_free(&row_comm);
```

这里使用原始的rank当作拆分操作的key，我们希望新通讯器中的所有进程与原始通讯器中的所有进程处于相同的顺序，因此在这里使用原始等级值最有意义，因为它已经正确地排序了。 之后，我们将打印出新的等级和大小以确保其有效。 输出应如下所示：

```cpp
WORLD RANK/SIZE: 0/16 	 ROW RANK/SIZE: 0/4
WORLD RANK/SIZE: 1/16 	 ROW RANK/SIZE: 1/4
WORLD RANK/SIZE: 2/16 	 ROW RANK/SIZE: 2/4
WORLD RANK/SIZE: 3/16 	 ROW RANK/SIZE: 3/4
WORLD RANK/SIZE: 4/16 	 ROW RANK/SIZE: 0/4
WORLD RANK/SIZE: 5/16 	 ROW RANK/SIZE: 1/4
WORLD RANK/SIZE: 6/16 	 ROW RANK/SIZE: 2/4
WORLD RANK/SIZE: 7/16 	 ROW RANK/SIZE: 3/4
WORLD RANK/SIZE: 8/16 	 ROW RANK/SIZE: 0/4
WORLD RANK/SIZE: 9/16 	 ROW RANK/SIZE: 1/4
WORLD RANK/SIZE: 10/16 	 ROW RANK/SIZE: 2/4
WORLD RANK/SIZE: 11/16 	 ROW RANK/SIZE: 3/4
WORLD RANK/SIZE: 12/16 	 ROW RANK/SIZE: 0/4
WORLD RANK/SIZE: 13/16 	 ROW RANK/SIZE: 1/4
WORLD RANK/SIZE: 14/16 	 ROW RANK/SIZE: 2/4
WORLD RANK/SIZE: 15/16 	 ROW RANK/SIZE: 3/4
```

## 5.3 其他通讯器创建函数

1. `MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)` ，它创建了一个通讯器的副本。对于使用库执行特殊函数的应用（例如数学库）非常有用。 在这类应用中，重要的是用户代码和库代码不要互相干扰。 为了避免这种情况，每个应用程序应该做的第一件事是创建 `MPI_COMM_WORLD` 的副本，这将避免其他使用 `MPI_COMM_WORLD` 的库的问题。 库本身也应该复制 `MPI_COMM_WORLD` 以避免相同的问题。
2. 另一个功能是`MPI_Comm_create`，函数原型是：
    
    ```cpp
    int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) 
    int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm)
    ```
    
    可以看到不同之处就是：`tag` 参数。`MPI_Comm_create_group` 仅是 `group` 中包含的一组进程的集合，而 `MPI_Comm_create` 是 `comm` 中每个进程的集合。
    

## 5.4 组的概念

刚刚出现的`Group`是更灵活的一种方式。在MPI内部，必须（除其他事项外）保持通讯器的两个主要部分，即区分一个通讯器与另一个通讯器的上下文（或 ID）以及该通讯器包含的一组进程。上下文阻止了与一个通讯器上的操作匹配的另一通讯器上的类似操作。 MPI 在内部为每个通讯器保留一个 ID，以防止混淆。

组更易于理解，因为它只是通讯器中所有进程的集合。对于 `MPI_COMM_WORLD`，这是由 `mpiexec` 启动的所有进程。 对于其他通讯器，组将有所不同。 在上面的示例代码中，组是所有以相同的 `color` 传参给 `MPI_Comm_split` 的进程。

如图：上面是并集，会从其他两个集合中创建一个新的（可能）更大的集合。 新集合包括前两个集合的所有成员（无重复）。下面是交集，会从其他两个集合中创建一个新的（可能）更小的集合。 新集合包括两个原始集合中都存在的所有成员。

![9](http://stxg6c3mb.hd-bkt.clouddn.com/image 8.png)

如上所述，通讯器包含一个上下文或 ID，以及一个组。 调用 `MPI_Comm_group` 会得到对该组对象的引用`MPI_Group* group`。

```cpp
MPI_Comm_group(
	MPI_Comm comm,
	MPI_Group* group)
```

通讯器内有上下文ID和`group`，所以`group`不能用来与其他`rank`通信，（因为它没有附加上下文）。但是可以获取组的秩和大小（分别为 `MPI_Group_rank` 和 `MPI_Group_size`）。

> 组特有的功能而通讯器无法完成的工作是：可以使用组在本地构建新的组。在此记住本地操作和远程操作之间的区别很重要。 远程操作涉及与其他秩的通信，而本地操作则没有。 创建新的通讯器是一项远程操作，因为所有进程都需要决定相同的上下文和组，而在本地创建组是因为它不用于通信，因此每个进程不需要具有相同的上下文。 您可以随意操作一个组，而无需执行任何通信。
> 

所以如果想让两组进程并起来或者交起来就会很容易，如下代码：

```cpp
MPI_Group_union(
	MPI_Group group1,
	MPI_Group group2,
	MPI_Group* newgroup)
MPI_Group_intersection(
	MPI_Group group1,
	MPI_Group group2,
	MPI_Group* newgroup)
```

现在再谈刚刚的`MPI_Comm_create_group`，这是一个用于创建新通讯器的函数，但无需像 `MPI_Comm_split` 之类那样需要进行计算以决定组成，该函数将使用一个 `MPI_Group` 对象并创建一个与组具有相同进程的新通讯器。

```cpp
int MPI_Comm_create_group(
    MPI_Comm comm,      // 输入的通信器（定义进程范围）
    MPI_Group group,    // 输入的组（定义新通信器包含的进程）
    int tag,            // 用于标识通信操作的标签（一般设置为 0）
    MPI_Comm *newcomm   // 输出的新通信器
);
```

more(该API允许您选择组中的特定秩并构建为新组):

```cpp
MPI_Group_incl(
	MPI_Group group,
	int n,
	const int ranks[],
	MPI_Group* newgroup)
```

# 6 整体框架

![last](http://stxg6c3mb.hd-bkt.clouddn.com/IMG_2766.heic)