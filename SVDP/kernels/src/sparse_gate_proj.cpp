#include <torch/extension.h>
#include <cuda_fp16.h>

void launch_sparse_gate_proj(__half *weight, __half *x, __half *prediction);

void torch_launch_sparse_gate_proj(torch::Tensor &weight, torch::Tensor &x, torch::Tensor &prediction) {
    launch_sparse_gate_proj((__half *)weight.data_ptr(), (__half *)x.data_ptr(), (__half *)prediction.data_ptr());
}

void register_kernel_gate(pybind11::module &m) {
    m.def("sparse_gate_proj", &torch_launch_sparse_gate_proj, "Sparse gate proj");
}