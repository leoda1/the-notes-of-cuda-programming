//包含用于归并排序的通用定义和辅助函数
////////////////////////////////////////////////////////////////////////////////
// Shortcut definitions
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;

#define SHARED_SIZE_LIMIT 1024U  //很多GPU一个块最多1024个线程
#define SAMPLE_STRIDE 128

////////////////////////////////////////////////////////////////////////////////
// Extensive sort validation routine
////////////////////////////////////////////////////////////////////////////////
extern "C" uint validateSortedKeys(uint *resKey, uint *srcKey, uint batchSize,
                                   uint arrayLength, uint numValues,
                                   uint sortDir);

extern "C" void fillValues(uint *val, uint N);

extern "C" int validateSortedValues(uint *resKey, uint *resVal, uint *srcKey,
                                    uint batchSize, uint arrayLength);

////////////////////////////////////////////////////////////////////////////////
// CUDA merge sort
////////////////////////////////////////////////////////////////////////////////
extern "C" void initMergeSort(void);

extern "C" void closeMergeSort(void);

extern "C" void mergeSort(uint *dstKey, uint *dstVal, uint *bufKey,
                          uint *bufVal, uint *srcKey, uint *srcVal, uint N,
                          uint sortDir);

////////////////////////////////////////////////////////////////////////////////
// CPU "emulation"
////////////////////////////////////////////////////////////////////////////////
extern "C" void mergeSortHost(uint *dstKey, uint *dstVal, uint *bufKey,
                              uint *bufVal, uint *srcKey, uint *srcVal, uint N,
                              uint sortDir);