#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}

/*
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                     aten::matmul         1.87%       1.884ms        69.25%      69.771ms      34.886ms       1.838ms         1.90%      71.701ms      35.850ms             2  
                                        aten::bmm        38.61%      38.900ms        63.92%      64.406ms      32.203ms      66.321ms        68.49%      66.321ms      33.160ms             2  
                                        aten::mul         2.16%       2.171ms        12.26%      12.351ms      12.351ms      12.891ms        13.31%      12.891ms      12.891ms             1  
                                    aten::softmax         0.30%     302.468us         8.50%       8.568ms       8.568ms     495.000us         0.51%       8.646ms       8.646ms             1  
                                   aten::_softmax         0.54%     548.259us         8.19%       8.249ms       8.249ms       8.151ms         8.42%       8.151ms       8.151ms             1  
                                  aten::transpose         2.89%       2.914ms         3.28%       3.303ms       3.303ms       3.086ms         3.19%       3.595ms       3.595ms             1  
                               aten::_unsafe_view         1.27%       1.281ms         1.27%       1.281ms     640.562us       1.364ms         1.41%       1.364ms     682.000us             2  
                                    aten::reshape         0.49%     492.091us         1.29%       1.300ms     324.947us     501.000us         0.52%       1.284ms     321.000us             4  
                                     aten::expand         0.76%     761.314us         0.80%     801.772us     200.443us     864.000us         0.89%     894.000us     223.500us             4  
                                 aten::as_strided         0.37%     373.007us         0.37%     373.007us      74.601us     539.000us         0.56%     539.000us     107.800us             5  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 100.753ms
Self CUDA time total: 96.833ms

=== profiling minimal flash attention === 
/home/joker/projects/cudalearn/notes/code/CUDA/1.7 FlashAttn/bench.py:38: FutureWarning: The attribute `use_cuda` will be deprecated soon, please use ``use_device = 'cuda'`` instead.
  with torch.autograd.profiler.profile(use_cuda=True) as prof:
Max shared memoty: 49152requset shared memory: 28672
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
          aten::zeros_like         0.29%      35.877us        68.37%       8.542ms       8.542ms       8.000us         0.08%       8.526ms       8.526ms             1  
               aten::zero_        13.86%       1.731ms        63.89%       7.982ms       3.991ms       2.124ms        21.20%       8.271ms       4.136ms             2  
               aten::fill_         9.67%       1.208ms        54.77%       6.843ms       3.421ms       6.669ms        66.57%       6.669ms       3.334ms             2  
                  aten::to         0.18%      22.268us         5.70%     712.312us     356.156us      20.000us         0.20%     748.000us     374.000us             2  
            aten::_to_copy         0.40%      50.427us         5.44%     679.664us     339.832us      63.000us         0.63%     728.000us     364.000us             2  
       aten::empty_strided         0.45%      55.937us         3.64%     454.165us     151.388us     549.000us         5.48%     549.000us     183.000us             3  
                aten::full         0.24%      29.546us         5.26%     657.499us     657.499us       8.000us         0.08%     531.000us     531.000us             1  
          aten::empty_like         3.96%     495.391us         4.26%     532.406us     532.406us     398.000us         3.97%     404.000us     404.000us             1  
               aten::zeros         0.26%      32.475us         0.56%      70.475us      70.475us      42.000us         0.42%     213.000us     213.000us             1  
               aten::copy_         0.28%      35.034us         1.26%     157.527us      78.764us     122.000us         1.22%     122.000us      61.000us             2  
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 12.494ms
Self CUDA time total: 10.018ms

attn values sanity check: True
*/