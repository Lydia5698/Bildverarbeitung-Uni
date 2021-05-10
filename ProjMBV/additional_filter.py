import SimpleITK as sitk
import numpy as np

import image_viewing as vis
from typing import List


def normalise(image: sitk.Image) -> sitk.Image:
    """ Normalises the image to (0, 500) and cuts of the edges.
    It will use the 5th and 99th percentile to cut of the edges.
    This method is only used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: normalised sitk.Image
    """

    stat_filter = sitk.StatisticsImageFilter()
    stat_filter.Execute(image)

    img_array = sitk.GetArrayFromImage(image)
    lower_percentile = np.percentile(img_array, 5)
    upper_percentile = np.percentile(img_array, 99)

    window_filter = sitk.IntensityWindowingImageFilter()
    window_filter.SetOutputMaximum(500)
    window_filter.SetOutputMinimum(0)
    window_filter.SetWindowMaximum(upper_percentile)
    window_filter.SetWindowMinimum(lower_percentile)
    return window_filter.Execute(image)


def gradient(image: sitk.Image) -> sitk.Image:
    """ Executes a filter to smoothe the image while keeping the edges.
    This method is only used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: smoothed sitk.Image
    """

    grad_filter = sitk.GradientAnisotropicDiffusionImageFilter()  #schon etwas zeitintensiv
    grad_filter.SetTimeStep(0.05)
    return grad_filter.Execute(sitk.Cast(image, sitk.sitkFloat32))


def seedpoints(image: sitk.Image) -> List[List[int]]:
    """ Gets seedpoints of the image.
    It will use every pixel where the intensity is above 490.
    This method is only used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: List of indexes of seedpoints.
    """
    img_arr = np.array(sitk.GetArrayFromImage(image))
    seeds = np.argwhere(img_arr > 490)
    return seeds.tolist()


def threshold(image: sitk.Image, seeds: List[List[int]]) -> sitk.Image:
    """ Computes a threshold filter on the image with the borders (200, 500).
    This method is only used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
            seeds (List[List[int]]): List of indices of the seedpoints
        Returns: thresholded sitk.Image
    """

    thresh_filter = sitk.ConnectedThresholdImageFilter()
    thresh_filter.SetSeedList(seeds)
    thresh_filter.SetLower(480)
    thresh_filter.SetUpper(500)
    return thresh_filter.Execute(image)


def labeling(image: sitk.Image) -> sitk.Image:
    """ Labels the image.
    This method is only used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: labeled sitk.Image
    """

    label_filter = sitk.BinaryImageToLabelMapFilter()
    label_filter.SetInputForegroundValue(1)
    return label_filter.Execute(image)


def connected_component(image: sitk.Image) -> sitk.Image:
    """ Searches for the biggest connected component in a labeled image.
    This method is only used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: sitk.Image with biggest label
    """

    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SortByObjectSizeOn()
    relabel_filter.Execute(image)
    if 1 not in sitk.GetArrayFromImage(image):
        return None
    biggest_label = relabel_filter.GetSizeOfObjectsInPixels()[0]
    if biggest_label < 20:      # keine Ahnung, was hier der beste Wert ist
        return None
    relabel_filter.SetMinimumObjectSize(biggest_label)
    return relabel_filter.Execute(image)


def opening(image: sitk.Image) -> sitk.Image:
    """ Performs an opening on a binary image.
    This method is accessible through pressing the o key and is used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: sitk.Image after opening
    """

    open_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    open_image = open_filter.Execute(image)
    sitk.WriteImage(open_image, 'segmentation.nii.gz')
    return open_image


def closing(image: sitk.Image) -> sitk.Image:
    """ Performs a closing on a binary image.
    This method is accessible through pressing the c key and is used by the segmentation pipeline.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: sitk.Image after closing
    """

    close_filter = sitk.BinaryMorphologicalClosingImageFilter()
    close_filter.SetKernelRadius((2,2,2))
    close_image = close_filter.Execute(image)
    sitk.WriteImage(close_image, 'segmentation.nii.gz')
    return close_image


def hole_filling(image: sitk.Image) -> sitk.Image:
    """ Filling holes in a binary image.
    This method is only accessible through pressing the f key.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: sitk.Image with less or smaller holes
    """

    hole_filter = sitk.VotingBinaryIterativeHoleFillingImageFilter()
    hole_filter.SetForegroundValue(1)
    hole_filter.SetBackgroundValue(0)
    hole_filter.SetRadius(1)
    hole_filter.SetMaximumNumberOfIterations(10)
    hole_image = hole_filter.Execute(image)
    sitk.WriteImage(hole_image, 'segmentation.nii.gz')
    return hole_image


def erode(image:sitk.Image) -> sitk.Image:
    """ Erode a binary image.
    This method is only accessible through pressing the e key.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: eroded sitk.Image
    """

    erode_filter = sitk.BinaryErodeImageFilter()
    erode_image = erode_filter.Execute(image)
    sitk.WriteImage(erode_image, 'segmentation.nii.gz')
    return erode_image


def dilate(image:sitk.Image) -> sitk.Image:
    """ Dilate a binary image.
    This method is only accessible through pressing the d key.
        Parameters:
            image (sitk.Image): brain-MRT image
        Returns: dilated sitk.Image
    """

    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_image = dilate_filter.Execute(image)
    sitk.WriteImage(dilate_image, 'segmentation.nii.gz')
    return dilate_image