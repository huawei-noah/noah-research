#include <torch/extension.h>
#include <cuda_fp16.h>


void launch_sparse_up_proj_drelu(__half *weight, __half *x, __half *gate_out, float threshold = 0.);

void torch_launch_sparse_up_proj_drelu(torch::Tensor &weight, torch::Tensor &x, torch::Tensor &gate_out, float threshold = 0.) {
    launch_sparse_up_proj_drelu((__half *)weight.data_ptr(),(__half *)x.data_ptr(), (__half *)gate_out.data_ptr(), threshold);
}

void register_kernel_up_drelu(pybind11::module &m) {
    m.def("sparse_up_proj_drelu", &torch_launch_sparse_up_proj_drelu, "Sparse up proj drelu");
}
