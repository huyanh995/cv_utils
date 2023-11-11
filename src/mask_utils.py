__author__ = 'Huy Anh Nguyen'

from typing import List, Union, Dict
from itertools import groupby
import numpy as np
from pycocotools import mask as mask_coco

def compute_iou(a: np.ndarray,
                b: np.ndarray) -> np.ndarray:

    raise NotImplementedError

def rle_encode(mask: np.ndarray) -> Dict[str, List[int]]:
    """
    Encode a binary mask using Run-Length Encoding (RLE).

    This function takes a binary mask as input and converts it into a run-length encoding format.
    Unlike pycocotools where return counts value is encoded as a bytestring, this function returns
    counts value as a list of integers.

    Source: https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset

    Parameters:
        mask (np.ndarray): A 2D binary numpy array representing the mask to be encoded. The mask must contain
                       only 0s and 1s, where 1 represents the object and 0 represents the background.

    Returns:
        Dict[str, List[int]]: A dictionary with two keys:
            - 'counts': A list of integers representing the lengths of runs of 0s and 1s in the mask,
                        starting with the count of 0s.
            - 'size': The dimensions of the mask as a list [height, width].

    Raises:
        AssertionError: If the input mask is not binary (i.e., does not contain exactly two unique values).

    Example:
    >>> mask = np.array([[0, 0, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1]])
    >>> rle_encode(mask)
    {'counts': [1, 1, 2, 1, 4, 1, 1, 2, 1, 1], 'size': [3, 5]}
    """

    labels = np.unique(mask)
    assert len(labels) == 2 and np.all(np.isin(labels, [0, 1])), \
        'Mask must be binary ([0, 1] or [True, False]), got {} unique values'.format(labels)

    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle_decode(rle: Dict[str, List[int]]) -> np.ndarray:
    """
    Decode a Run-Length Encoded (RLE) mask into a binary mask array.

    This function takes a dictionary representing the RLE of a mask and decodes it into a 2D binary numpy array.
    The RLE format should include 'counts' representing the lengths of runs of 0s and 1s in the mask and 'size'
    representing the dimensions of the mask.

    Parameters:
        rle (Dict[str, List[int]]): A dictionary with two keys:
            - 'counts': A list of integers representing the lengths of runs of 0s and 1s in the mask.
            - 'size': The dimensions of the mask as a list [height, width].

    Returns:
        np.ndarray: A 2D binary numpy array representing the decoded mask. The shape of the array is given by 'size' in the RLE.

    Example:
    >>> rle = {'counts': [1, 2, 1, 2, 4, 1, 1, 2], 'size': [3, 3]}
    >>> rle_decode(rle)
    array([[0, 1, 1],
           [1, 0, 0],
           [0, 0, 1]], dtype=uint8)

    Note:
    The function relies on 'pycocotools' for decoding. Ensure this package is installed and imported as 'mask_coco'.
    """
    rle_obj = mask_coco.frPyObjects(rle, *rle['size'])
    return mask_coco.decode(rle_obj)

if __name__ == '__main__':
    mask = np.array([[False, False, False, True, True], [True, True, True, False, False], [False, False, False, True, True]])
    rle = rle_encode(mask)
    print(f'Encoded: {rle}')

    mask = rle_decode(rle)
    print(f'Decoded: {mask}')

