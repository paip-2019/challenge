from tifffile import imsave
import skimage.io as io

# an example filename
submission_filename = 'case_083.tif'
# print(fn)

# load image: any library you're familiar with
img = io.imread(fn)

# save with compression: ADOBE_DEFLATE algorithm with level 9 (among 0~9)
# NOTE: the saving filename is the same with the loading filename. In other words, it may overwrite your original image.
imsave(fn, img, compress=9)
