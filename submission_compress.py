from tifffile import imsave
import skimage.io as io
import numpy as np

# an example filename
submission_filename = 'case_083.tif'
# print(submission_filename)

# load image: any library you're familiar with
img = io.imread(submission_filename)
img = img.astype(np.uint8) # NOTE: tifffile.imsave would show a 'bilevel'-related error w/o this explicit type casting

# save with compression: ADOBE_DEFLATE algorithm with level 9 (among 0~9)
# NOTE: the saving filename is the same with the loading filename. In other words, it may overwrite your original image.
imsave(submission_filename, img, compress=9)
