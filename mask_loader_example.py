import skimage.io as io
import numpy as np
import os
from __future__ import print_function

def mask_loader(fn):
  assert(os.path.isfile(fn))
  mask = io.imread(fn)
  print('mask shape:', mask.shape)
  return mask

if __name__ == '__main__':
  load_fn = 'sample_mask.tif'
  save_fn = 'decompressed_mask.tif'

  mask = mask_loader(load_fn)
  io.imsave(mask, save_fn)

