import SimpleITK as sitk
import sys
import image_viewing as vis
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv) > 1, 'No input image specified!'

# load example image from argv
input_image = sitk.ReadImage(sys.argv[1])

# load segmentation to get dice
seg_image = sitk.ReadImage(sys.argv[2])


#vis.show_image(input_image, 'input', False)

# normalise image to [0,500] and remove measurement errors between (min, min+5) and (max-5,max)
stat_filter = sitk.StatisticsImageFilter()
stat_filter.Execute(input_image)
max = stat_filter.GetMaximum()
min = stat_filter.GetMinimum()
window_filter = sitk.IntensityWindowingImageFilter()
window_filter.SetOutputMaximum(500)
window_filter.SetOutputMinimum(0)
window_filter.SetWindowMaximum(max - 5)
window_filter.SetWindowMinimum(min + 5)
normalised_image = window_filter.Execute(input_image)
vis.show_image(normalised_image, 'normalised', False)

# image smoothing with edge preservation
grad_filter = sitk.GradientAnisotropicDiffusionImageFilter()  #schon etwas zeitintensiv
grad_filter.SetTimeStep(0.05)
grad_image = grad_filter.Execute(sitk.Cast(normalised_image, sitk.sitkFloat32))
#vis.show_image(grad_image, 'grad', False)


# set seedpoints manually
#seeds = vis.show_and_return_markers(grad_image, 'Set Seedpoints')


#get every 5000th pixel as seedpoint
arr = np.array(sitk.GetArrayViewFromImage(grad_image))
seeds = []
for idx, i in np.ndenumerate(arr):
    if i % 5000 == 0:
        seeds.append(idx)


# compute thresh image with lower = 200, upper = 500 for all seedpoints
thresh_filter = sitk.ConnectedThresholdImageFilter()
thresh_filter.SetSeedList(seeds)
thresh_filter.SetLower(200)
thresh_filter.SetUpper(500)
thresh_image = thresh_filter.Execute(grad_image)
#vis.show_image(thresh_image, 'thresh', True)

# opening and closing on the image
open_filter = sitk.BinaryMorphologicalOpeningImageFilter()
open_image = open_filter.Execute(thresh_image)
close_filter = sitk.BinaryMorphologicalClosingImageFilter()
close_image = close_filter.Execute(open_image)
#vis.show_image(close_image,'opening&closing', False)

# label the connected components
label_filter = sitk.BinaryImageToLabelMapFilter()
label_filter.SetInputForegroundValue(1)
label_image = label_filter.Execute(close_image)

# choose the biggest connected component
relabel_filter = sitk.RelabelComponentImageFilter()
relabel_filter.SortByObjectSizeOn()
relabel_filter.Execute(close_image)
relabel_filter.SetMinimumObjectSize(relabel_filter.GetSizeOfObjectsInPixels()[0])
relabel_image = relabel_filter.Execute(close_image)
#vis.show_image(relabel_image, 'segmentation', True)

# get changes made by key-actions
#relabel_image = sitk.ReadImage('segmentation.nii.gz')

# show the input_image with the segmentation
vis.show_image_with_mask(input_image, relabel_image, 'segmentation with image', 'b', False)

# get the dice from our segmentation and given segmentation
measures = sitk.LabelOverlapMeasuresImageFilter()
measures.Execute(relabel_image,seg_image)
print("Dice: ", measures.GetDiceCoefficient())

# get changes after further key-actions
