from nms_gpu  import rotate_iou_gpu_eval
import numpy as np

def main():
  boxes = np.array([
    [0, 0, 1,   2., 0.1],
    [0, 0, .001,   2., 0.1],
    [0, 0, 0.1, 2., 0.5],
    [0, 0, 0.1, 2., -np.pi/2],
    ])

  ious = np.diag( rotate_iou_gpu_eval(boxes, boxes) )
  print(f"ious: {ious}")
  #old: [0.         0.         0.33333316 0.        ]
  #new: [1.         0.99998605 0.99999934 1.        ]

if __name__ == '__main__':
  main()
