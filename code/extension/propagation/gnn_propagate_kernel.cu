#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cuda_fp16.h>

__global__ void gnn_propagate_forward_kernel(int* initial_rank, at::Half* A, at::Half* A_qe, at::Half* S, const int sample_num, const int topk, const int total_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < total_num; i += stride) {
        int sample_index = i / sample_num;
        int fea = i % sample_num;
        at::Half sum = 0.0;
        for (int j = 0; j < topk ; j++) {
            int select_index = sample_index*topk+j;
            int topk_fea_index = initial_rank[select_index] * sample_num + fea;
            sum += A[topk_fea_index] * S[select_index];
        }
        A_qe[i] = sum;
    }
}


at::Tensor gnn_propagate_forward(at::Tensor A, at::Tensor initial_rank, at::Tensor S) {
    const auto sample_num = A.size(0);
    const auto topk = initial_rank.size(1);

    const auto total_num = sample_num * sample_num ;
    auto A_qe = torch::zeros({sample_num, sample_num}, at::device(initial_rank.device()).dtype(at::ScalarType::Half)); // 使用半精度

    const int threads = 256;
    const int blocks = (total_num + threads - 1) / threads;

    gnn_propagate_forward_kernel<<<blocks, threads>>>(initial_rank.data_ptr<int>(), A.data_ptr<at::Half>(), A_qe.data_ptr<at::Half>(), S.data_ptr<at::Half>(), sample_num, topk, total_num);
    return A_qe;
}