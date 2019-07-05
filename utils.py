from __future__ import print_function
import skimage.io as io
from skimage.transform import resize
import numpy as np
import os

OVERLAY_MASK_RATIO = 0.3

def mask_loader(fn):
  '''
  This is a simplest loader for the given tif mask labels, which are compressed in 'LZW' format for logistic convenience.
  Scikit-image library can automatically decompress and load them on your physical memory.
  '''
  assert(os.path.isfile(fn))
  mask = io.imread(fn)
  print('mask shape:', mask.shape)
  return mask

def gen_overlay(orig_img, msk):
  '''
  We don't give a loader for original svs image because there are well-known open source libraries already.
  (e.g. openslide, pyvips, etc.)
  We assume that original image has [H, W, C(=3)] dimension and mask has [H, W] dimension.
  '''
  assert(orig_img.shape[:-1] == msk.shape)
  img_dark = (orig_img * (1.0 - OVERLAY_MASK_RATIO)).astype(np.uint8)
  gmsk = np.zeros(orig_img.shape, dtype=np.uint8)
  gmsk[:, :, 1] += (msk * 255 * OVERLAY_MASK_RATIO).astype(np.uint8) # assign GREEN color for mask labels
  img_dark += gmsk
  img_dark[img_dark>255] = 255
  return img_dark.astype(np.uint8)

def fit_viewer(img):
  assert(img.ndim == 2 or img.ndim == 3)
  limit = 2147483647
  if img.size > limit:
    ratio = np.sqrt(limit/img.size) * 0.99
    if img.ndim > 2:
      ratio /= img.shape[2]
    target_shape = [int(img.shape[0]*ratio), int(img.shape[1]*ratio)]
    if img.ndim > 2:
      target_shape.append(3)
    img_resized = resize(img, target_shape, preserve_range=True) # it may take a lot of memory space
    return img_resized
  else:
    return img

if __name__ == '__main__':
  '''
  It is not guaranteed that this sample code will work seamlessly,
    nor optimized for obtaining results as quick as possible.
  We hope you just can find some useful tips from here.

  '''

  # Trying to save a decompressed tif of a mask label.

  mask = mask_loader('a_sample_mask.tif')
  io.imsave('a_decompressed_mask.tif', mask)

  ## =================================================

  # Trying to save an overlay image between original svs image and mask image.
  # Please note that you need to load an original image on pysical memory before this task.

  #orig_img = your_loader('filename.svs')
  overlay = gen_overlay(orig_img, mask)
  io.imsave('a_sample_overlay.tif', overlay)

  ## =================================================

  # Trying to make an image fit into simple viewers(e.g. Windows default, Fiji, etc.),
  #   which only can handle a smaller number of elements than 2^31.
  # When the size of the original overlay exceeds the capable level, you may need this code for resizing.

  overlay_fit = fit_viewer(overlay)
  io.imsave('a_sample_overlay_fit.tif', overlay)

  ## =================================================


