import SimpleITK as sitk
import sys
import image_viewing as vis
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import additional_filter as filter


assert len(sys.argv) > 1, 'No input image specified!'
# load example image from argv
path = 'ISLES2015_Train/'+sys.argv[1]+'/VSD.Brain.'+sys.argv[1]+'.O.MR_Flair_reg.nii.gz'
input_image = sitk.ReadImage(path)
#vis.show_image(input_image, 'input', False)

# normalise image to [0,500] and remove measurement errors below the 5th and above the 99th percentile
normalised_image = filter.normalise(input_image)
#vis.show_image(normalised_image, 'Flair normalised', False)

# image smoothing with edge preservation
grad_image = filter.gradient(normalised_image)
#vis.show_image(grad_image, 'Flair grad', False)

# get seedpoints
seeds = filter.seedpoints(grad_image)

# compute thresh image with lower = 200, upper = 500 for all seedpoints
thresh_image = filter.threshold(grad_image, seeds)
#vis.show_image(thresh_image, 'thresh', False)
#vis.show_image_with_mask(normalised_image, thresh_image)

# opening and closing on the image
open_image = filter.opening(thresh_image)
close_image = filter.closing(open_image)
dilate_image = filter.dilate(close_image)
hole_image = filter.hole_filling(dilate_image)
#vis.show_image(hole_image, 'Flair opening&closing&dilate&hole', False)

# label the connected components
label_image = filter.labeling(hole_image)


# choose the biggest connected component
cast = sitk.CastImageFilter()
cast.SetOutputPixelType(sitk.sitkInt32)
relabel_image = filter.connected_component(cast.Execute(label_image))
# test, if there is any label selected and if it is big enough.
if relabel_image is not None:
    sitk.WriteImage(relabel_image, 'segmentation.nii.gz')
    #vis.show_image(relabel_image, 'Flair segmentation', True)

    # get changes made by key-actions
    relabel_image = sitk.ReadImage('segmentation.nii.gz')

    # show the input_image with the segmentation
    vis.show_image_with_mask(normalised_image, relabel_image, 'Flair segmentation with image', 'b', False)
else: # use DWI image
    # load example image from argv
    path = 'ISLES2015_Train/' + sys.argv[1] + '/VSD.Brain.' + sys.argv[1] + '.O.MR_DWI_reg.nii.gz'
    input_image = sitk.ReadImage(path)

    # normalise image to [0,500] and remove measurement errors between (min, min+5) and (max-5,max)
    normalised_image = filter.normalise(input_image)
    #vis.show_image(normalised_image, 'DWI normalised', False)

    # image smoothing with edge preservation
    grad_image = filter.gradient(normalised_image)
    #vis.show_image(grad_image, 'grad', False)

    # get seedpoints
    seeds = filter.seedpoints(grad_image)

    # compute thresh image with lower = 200, upper = 500 for all seedpoints
    thresh_filter = sitk.ConnectedThresholdImageFilter()
    thresh_filter.SetSeedList(seeds)
    thresh_filter.SetLower(470)
    thresh_filter.SetUpper(500)
    thresh_image = thresh_filter.Execute(grad_image)
    #vis.show_image(thresh_image, 'DWI thresh', True)

    # opening and closing on the image
    open_image = filter.opening(thresh_image)
    close_image = filter.closing(open_image)
    hole_image = filter.hole_filling(close_image)
    #vis.show_image(close_image, 'DWI opening&closing', False)

    # label the connected components
    label_image = filter.labeling(hole_image)

    # choose the biggest connected component
    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkInt32)
    relabel_image = filter.connected_component(cast.Execute(label_image))
    sitk.WriteImage(relabel_image, 'segmentation.nii.gz')
    vis.show_image(relabel_image, 'DWI segmentation', True)

    # get changes made by key-actions
    relabel_image = sitk.ReadImage('segmentation.nii.gz')

    # show the input_image with the segmentation
    vis.show_image_with_mask(normalised_image, relabel_image, 'DWI segmentation with image', 'b', False)


# load given segmentation
path = 'ISLES2015_Train/'+sys.argv[1]+'/VSD.Brain.'+sys.argv[1]+'.O.OT_reg.nii.gz'
seg_image = sitk.ReadImage(path)
# cast segmentation
cast = sitk.CastImageFilter()
cast.SetOutputPixelType(sitk.sitkInt32)
casted_seg_image = cast.Execute(seg_image)

# calculate Dice from our segmentation and given segmentation
measures = sitk.LabelOverlapMeasuresImageFilter()
measures.Execute(relabel_image, casted_seg_image)

hausdorff_Distance = sitk.HausdorffDistanceImageFilter()
hausdorff_Distance.Execute(relabel_image, casted_seg_image)

print("Dice: ", measures.GetDiceCoefficient())
print("Jaccard: ", measures.GetJaccardCoefficient())
print("Hausdorff: ", hausdorff_Distance.GetHausdorffDistance())

#calculate the volume of the segmentation
shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.Execute(relabel_image)

for i in shape_stats.GetLabels():
    size = shape_stats.GetPhysicalSize(i)
print("Volume: ", size, "mm^3")

vis.show_image_with_mask(input_image, seg_image, 'reference segmentation with image', 'b', False)
