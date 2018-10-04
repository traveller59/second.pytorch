#include "nms.h"
#include "nms_cpu.h"
PYBIND11_MODULE(nms, m)
{
    m.doc() = "non_max_suppression asd";
    m.def("non_max_suppression", &non_max_suppression<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression", &non_max_suppression<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression_cpu", &non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("non_max_suppression_cpu", &non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
}