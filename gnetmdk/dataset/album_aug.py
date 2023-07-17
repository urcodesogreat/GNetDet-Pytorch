import functools

import cv2
import torch
import random
import numpy as np
import albumentations as A  # pip install -U albumentations --no-binary imgaug,albumentations

# In some systems, in the multiple GPU regime, PyTorch may deadlock the DataLoader if
# OpenCV was compiled with OpenCL optimizations. Adding the following two lines before
# the library import may help. For more details [https://github.com/pytorch/pytorch/issues/1355]
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Transformation:
    """
    Image augmentation using `Albumentations` library.

    Refer to [Link](https://albumentations.ai) for official documents.
    Refer to [Link](https://github.com/albumentations-team/albumentations#list-of-augmentations)
             for full set of APIs.
    Refer to [Link](https://albumentations-demo.herokuapp.com/) for online demo.
    """

    @staticmethod
    def random_blur(p: float = 0.5):
        """
        Randomly choose one of the following method to blur the input image.
        """
        return A.OneOf([
            # Blur the input image using a random-sized kernel
            A.Blur(blur_limit=5),
            # Apply glass noise to the input image
            A.GlassBlur(),
            # Blur the input image using a Gaussian filter with a random kernel size
            A.GaussianBlur(),
            # Blur the input image using a median filter with a random aperture linear size
            A.MedianBlur(),
            # Apply motion blur to the input image using a random-sized kernel
            A.MotionBlur(blur_limit=9),
        ], p=p)

    @staticmethod
    def random_noise(p: float = 0.5):
        """
        Randomly choose one of the following method to add noise to the input image.
        """
        return A.OneOf([
            # Apply camera sensor noise.
            A.ISONoise(),
            # Apply gaussian noise to the input image
            A.GaussNoise(),
            # Multiply image to random number or array of numbers
            A.MultiplicativeNoise()
        ], p=p)

    @staticmethod
    def random_quality(p: float = 0.5):
        """
        Randomly decreases image quality.
        """
        return A.OneOf([
            # Decreases image quality by downscaling and upscaling back.
            A.Downscale(0.5, 0.99, interpolation=cv2.INTER_CUBIC),
            # Decrease Jpeg compression of an image.
            A.ImageCompression(25, 100, )
        ], p=p)

    @staticmethod
    def color_jitter(p: float = 0.5):
        """
        Randomly change color of input image.
        """
        return A.OneOf([
            A.ColorJitter(),
        ])


    @staticmethod
    def random_cutout(image, *, p: float = 0.5):
        """
        CoarseDropout of the square regions in the image.
        """
        max_holes = random.randint(5, 12)
        max_h = random.randint(8, 16)
        max_w = random.randint(8, 16)
        return A.CoarseDropout(max_holes, max_h, max_w, p=p)(image=image)


class Mosaic:
    """
    Mosaic image augmentation.
    """
    def __init__(self, ):
        pass


if __name__ == '__main__':
    trans = Transformation()
