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
# vis.show_image(input_image, 'input', False)

# normalise image to [0,500] and remove measurement errors between (min, min+5) and (max-5,max)
normalised_image = filter.normalise(input_image)
vis.show_image(normalised_image, 'Flair normalised', False)

# image smoothing with edge preservation
grad_image = filter.gradient(normalised_image)
vis.show_image(grad_image, 'Flair grad', False)

# get seedpoints
seeds = filter.seedpoints(grad_image)

# compute thresh image with lower = 200, upper = 500 for all seedpoints
thresh_image = filter.threshold(grad_image, seeds)
# vis.show_image(thresh_image, 'thresh', True)

# opening and closing on the image
open_image = filter.opening(thresh_image)
close_image = filter.closing(open_image)
vis.show_image(close_image, 'Flair opening&closing', False)

# label the connected components
label_image = filter.labeling(close_image)
# test, if there is any label selected


# choose the biggest connected component
cast = sitk.CastImageFilter()
cast.SetOutputPixelType(sitk.sitkInt32)
relabel_image = filter.connected_component(cast.Execute(label_image))
if relabel_image is not None:
    sitk.WriteImage(relabel_image, 'segmentation.nii.gz')
    vis.show_image(relabel_image, 'Flair segmentation', True)

    # get changes made by key-actions
    relabel_image = sitk.ReadImage('segmentation.nii.gz')

    # show the input_image with the segmentation
    vis.show_image_with_mask(input_image, relabel_image, 'Flair segmentation with image', 'b', False)
else:
    # load example image from argv
    path = 'ISLES2015_Train/' + sys.argv[1] + '/VSD.Brain.' + sys.argv[1] + '.O.MR_DWI_reg.nii.gz'
    input_image = sitk.ReadImage(path)

    # normalise image to [0,500] and remove measurement errors between (min, min+5) and (max-5,max)
    normalised_image = filter.normalise(input_image)
    vis.show_image(normalised_image, 'DWI normalised', False)

    # image smoothing with edge preservation
    grad_image = filter.gradient(normalised_image)
    # vis.show_image(grad_image, 'grad', False)

    # get seedpoints
    seeds = filter.seedpoints(grad_image)

    # compute thresh image with lower = 200, upper = 500 for all seedpoints
    thresh_filter = sitk.ConnectedThresholdImageFilter()
    thresh_filter.SetSeedList(seeds)
    thresh_filter.SetLower(180)
    thresh_filter.SetUpper(500)
    thresh_image = thresh_filter.Execute(grad_image)
    # thresh_image = filter.threshold(grad_image, seeds)
    # vis.show_image(thresh_image, 'DWI thresh', True)

    # opening and closing on the image
    open_image = filter.opening(thresh_image)
    close_image = filter.closing(open_image)
    vis.show_image(close_image, 'DWI opening&closing', False)

    # label the connected components
    label_image = filter.labeling(close_image)

    # choose the biggest connected component
    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkInt32)
    relabel_image = filter.connected_component(cast.Execute(label_image))
    sitk.WriteImage(relabel_image, 'segmentation.nii.gz')
    vis.show_image(relabel_image, 'DWI segmentation', True)

    # get changes made by key-actions
    relabel_image = sitk.ReadImage('segmentation.nii.gz')

    # show the input_image with the segmentation
    vis.show_image_with_mask(input_image, relabel_image, 'DWI segmentation with image', 'b', False)


#load given segmentation
path = 'ISLES2015_Train/'+sys.argv[1]+'/VSD.Brain.'+sys.argv[1]+'.O.OT_reg.nii.gz'
seg_image = sitk.ReadImage(path)

#calculate Dice from our segmentation and given segmentation
measures = sitk.LabelOverlapMeasuresImageFilter()
hausdorff_Distance = sitk.HausdorffDistanceImageFilter()

measures.Execute(relabel_image, seg_image)
hausdorff_Distance.Execute(seg_image, relabel_image)
print("Dice: ", measures.GetDiceCoefficient())
print("Jaccard: ", measures.GetJaccardCoefficient())
print("Hausdorff: ", hausdorff_Distance.GetHausdorffDistance())

vis.show_image_with_mask(input_image, seg_image, 'reference segmentation with image', 'b', False)

# vor Änderung des Thresholds
# 01 Hausdorff 12.0, Jaccard 0.6186 Dice 0.7644
# 02 Hausdorff 24.5153, Jaccard 0.6357 Dice 0.7773
# 05 Hausdorff 87.5957, Jaccard 0.2921 Dice 0.4521
# 09 Hausdorff 47.1381, Jaccard 0.6688, Dice 0.8015
# 10 War sehr dunkel funktioniert nicht mit eigenen Seedpoints Mit DWI Bild probiert:
#    Hausdorff 34.1906, Jaccard 0.1585, Dice 0.2736
# 12 Hausdorff 84.3030, Jaccard 0.0648, Dice 0.1217
# 15 Hausdorff 61.0082, Jaccard 0.4768, Dice 0.6457
# 22 War sehr dunkel funktioniert nicht mit eigenen Seedpoints
