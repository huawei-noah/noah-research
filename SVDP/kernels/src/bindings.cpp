#include <torch/extension.h>

void register_kernel_gate(pybind11::module &);
void register_kernel_up(pybind11::module &);
void register_kernel_up_drelu(pybind11::module &);
void register_kernel_down(pybind11::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_kernel_gate(m);
    register_kernel_up(m);
    register_kernel_up_drelu(m);
    register_kernel_down(m);
}