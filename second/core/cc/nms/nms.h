#ifndef NMS_H
#define NMS_H
#include <pybind11/pybind11.h>
// must include pybind11/stl.h if using containers in STL in arguments.
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

template <typename DType, int BLOCK_THREADS>
int _nms_gpu(int *keep_out, const DType *boxes_host, int boxes_num,
          int boxes_dim, DType nms_overlap_thresh, int device_id);

constexpr int const threadsPerBlock = sizeof(unsigned long long) * 8;

namespace py = pybind11;
using namespace pybind11::literals;
template <typename DType>
int non_max_suppression(
    py::array_t<DType> boxes,
    py::array_t<int> keep_out,
    DType nms_overlap_thresh,
    int device_id)
{
  py::buffer_info info = boxes.request();
  auto boxes_ptr = static_cast<DType *>(info.ptr);
  py::buffer_info info_k = keep_out.request();
  auto keep_out_ptr = static_cast<int *>(info_k.ptr);
  
  return _nms_gpu<DType, threadsPerBlock>(keep_out_ptr, boxes_ptr, boxes.shape(0), boxes.shape(1), nms_overlap_thresh, device_id);

}
#endif