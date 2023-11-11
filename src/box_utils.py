__author__ = 'Huy Anh Nguyen'

from typing import List, Union, TypeVar
import numpy as np

BoxType = Union[List[Union[float, int]], np.ndarray]

def compute_iou(a: Union[List[Union[float, int]], np.ndarray],
                b: Union[List[Union[float, int]], np.ndarray]) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    This function supports both individual and batch processing. For individual bounding boxes,
    inputs should be two bounding boxes represented as [x1, y1, x2, y2], where (x1, y1) is the
    top-left corner, and (x2, y2) is the bottom-right corner. For batch processing, inputs
    should be two arrays of shape (N, 4) and (M, 4) respectively, representing N and M bounding boxes.

    Parameters:
        a (Union[List[Union[float, int]], np.ndarray]): The first bounding box or an array of bounding boxes.
        b (Union[List[Union[float, int]], np.ndarray]): The second bounding box or an array of bounding boxes.

    Returns:
        np.ndarray: An array of shape (N, M) containing the IoU values. Each element [i, j] in the array
                    represents the IoU of the i-th bounding box in 'a' with the j-th bounding box in 'b'.

    Example:
    >>> compute_iou([1, 1, 3, 3], [2, 2, 4, 4])
    array([[0.14285714]])

    >>> compute_iou([[1, 1, 3, 3], [2, 2, 5, 5]], [[2, 2, 4, 4]])
    array([[0.14285714],
           [0.44444444]])

    >>> a = np.array([[1, 1, 3, 3], [4, 4, 6, 6]])
    >>> b = np.array([[2, 2, 4, 4], [5, 5, 7, 7]])
    >>> compute_iou(a, b)
    array([[0.14285714, 0.        ],
           [0.        , 0.14285714]])
    """

    a = np.array(a)
    b = np.array(b)

    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    # Compute pairwise intersection coordinates between boxes
    [xmin_a, ymin_a, xmax_a, ymax_a] = np.split(a, 4, axis=1)
    [xmin_b, ymin_b, xmax_b, ymax_b] = np.split(b, 4, axis=1)

    min_ymax_pairs = np.minimum(ymax_a, ymax_b.T) # Pairwise minimum of ymax a and b
    max_ymin_pairs = np.maximum(ymin_a, ymin_b.T)
    intersection_heights = np.maximum(0, min_ymax_pairs - max_ymin_pairs)

    min_xmax_pairs = np.minimum(xmax_a, xmax_b.T)
    max_xmin_pairs = np.maximum(xmin_a, xmin_b.T)
    intersection_widths = np.maximum(0, min_xmax_pairs - max_xmin_pairs)

    intersections_area = intersection_heights * intersection_widths

    # Computer pairwise union areas
    a_areas = (ymax_a - ymin_a) * (xmax_a - xmin_a)
    b_areas = (ymax_b - ymin_b) * (xmax_b - xmin_b)
    unions_area = a_areas + b_areas.T - intersections_area

    # Compute IoU
    iou = intersections_area / unions_area

    return iou

def xyxy_to_xywh(box: BoxType) -> BoxType:
    """
    Convert a bounding box from (x1, y1, x2, y2) to (x, y, width, height) format.

    This function accepts a single bounding box as a list or a numpy array. It converts
    the coordinates from top-left and bottom-right points (x1, y1, x2, y2) to top-left
    point with width and height (x, y, width, height). The output format matches the input format.

    Args:
        box (BoxType): A bounding box or an array of bounding boxes.

    Returns:
        BoxType: The bounding box(es) in (x, y, width, height) format.
    """

    is_list = isinstance(box, list)

    # Check for single box case
    if is_list and all(isinstance(coord, (int, float)) for coord in box):
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    box = np.array(box, dtype=np.float32)
    if box.ndim == 1:
        box = box[np.newaxis, :]

    box[:, 2] -= box[:, 0]
    box[:, 3] -= box[:, 1]

    return box.tolist() if is_list else box

def xywh_to_xyxy(box: BoxType) -> BoxType:
    """
    Convert a bounding box from (x, y, width, height) to (x1, y1, x2, y2) format.

    This function accepts a single bounding box as a list or a numpy array. It converts
    the coordinates from top-left point with width and height (x, y, width, height) to
    top-left and bottom-right points (x1, y1, x2, y2). The output format matches the input format.

    Args:
        box (BoxType): A bounding box or an array of bounding boxes.

    Returns:
        BoxType: The bounding box(es) in (x1, y1, x2, y2) format.
    """
    is_list = isinstance(box, list)

    # Check for single box case
    if is_list and all(isinstance(coord, (int, float)) for coord in box):
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

    box = np.array(box, dtype=np.float32)
    if box.ndim == 1:
        box = box[np.newaxis, :]

    box[:, 2] += box[:, 0]  # Convert width to x2
    box[:, 3] += box[:, 1]  # Convert height to y2

    return box.tolist() if is_list else box

def xyxy_to_xyah():
    raise NotImplementedError

def xyxy_to_cxcywh():
    raise NotImplementedError

if __name__ == '__main__':
    print(compute_iou([1, 1, 3, 3], [2, 2, 4, 4]))
    print(compute_iou([[1, 1, 3, 3], [2, 2, 5, 5]], [[2, 2, 4, 4]]))
    a = np.array([[1, 1, 3, 3], [4, 4, 6, 6]])
    b = np.array([[2, 2, 4, 4], [5, 5, 7, 7]])
    print(compute_iou(a, b))

    bbox = [1, 1, 3, 3]
    print(xyxy_to_xywh(bbox))

    bbox = [[1, 1, 3, 3], [4, 4, 6, 6]]
    print(xyxy_to_xywh(bbox), type(xyxy_to_xywh(bbox)))

    bbox = np.array([[1, 1, 3, 3], [4, 4, 6, 6]])
    print(xyxy_to_xywh(bbox), type(xyxy_to_xywh(bbox)))
