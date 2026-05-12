#include <torch/extension.h>
#include <cuda_fp16.h>


void launch_sparse_down_proj(__half *weight, __half *x, __half *res);

void torch_launch_sparse_down_proj(torch::Tensor &weight, torch::Tensor &x, torch::Tensor &res) {
    launch_sparse_down_proj((__half *)weight.data_ptr(), (__half *)x.data_ptr(), (__half *)res.data_ptr());
}

void register_kernel_down(pybind11::module &m) {
    m.def("sparse_down_proj", &torch_launch_sparse_down_proj, "Sparse down proj");
}