import numpy as np
import skimage.io as io
from threading import Thread
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from scipy.ndimage import label, binary_fill_holes
from tqdm import tqdm
from skimage.measure import regionprops
import os, glob, warnings, pyvips, tifffile
warnings.filterwarnings('ignore', category=UserWarning)
from xml_to_mask import xml2mask # NOTE

w_size = 256
stride = 256

total = w_size**2
min_px = total // 10
rm_min_size = 10
rm_min_hole_size = 10
pad_px = w_size * 2
max_rgb = [235, 210, 235]
threshold_vote = 0

def extract_coord(coord_str):
  y_start = coord_str.split('x')[0]
  x_start = coord_str.split('x')[1]
  return int(y_start), int(x_start)

def find_minmax(img):
  rps = regionprops(img)
  assert(len(rps) == 1)
  rp = rps[0]
  y_min, x_min, y_max, x_max = rp.bbox
  y_min = max(y_min - pad_px, 0)
  y_max = min(y_max + pad_px, img.shape[0])
  x_min = max(x_min - pad_px, 0)
  x_max = min(x_max + pad_px, img.shape[1])
  return (y_min, x_min, y_max, x_max)

def unit_threshold_and_amend(single_ch_img, threshold, ret):
  temp = single_ch_img < threshold
  temp = remove_small_holes(label(temp)[0], area_threshold=rm_min_hole_size)>0
  #temp = binary_fill_holes(temp)
  temp = remove_small_objects(label(temp)[0], min_size=rm_min_size)>0
  ret += temp #myutils.multiply(temp, ret>0) # stacking

def threshold_and_amend(img, ret):
  board = np.zeros(img.shape[:-1], dtype=np.uint8)
  threads = []
  for i in range(3):
    t = Thread(target=unit_threshold_and_amend, args=(img[:, :, i], max_rgb[i], board))
    threads.append(t)
    t.start()
  for t in threads:
    t.join()
  ret += (board>threshold_vote).astype(np.uint8)

def find_foreground(img):
  threshold_tissue = np.zeros(img.shape[:-1], dtype=np.uint8)
  #entropy_tissue = np.zeros(img.shape[:-1], dtype=np.uint8)
  threads = []
  t = Thread(target=threshold_and_amend, args=(img, threshold_tissue))
  threads.append(t)
  t.start()
  #t = Thread(target=entropy_and_amend, args=(img, entropy_tissue))
  #threads.append(t)
  #t.start()
  for t in threads:
    t.join()
  # threshold_tissue_only_big = remove_small_objects(label(threshold_tissue)[0], min_size=256**2)>0
  #tissue = ((threshold_tissue + entropy_tissue) > threshold_vote).astype(np.uint8)
  tissue = (threshold_tissue > threshold_vote).astype(np.uint8)
  return tissue
  # return myutils.multiply(threshold_tissue, entropy_tissue).astype(np.uint8)
  #return threshold_tissue

# https://github.com/libvips/pyvips/blob/master/examples/pil-numpy-pyvips.py
# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


if __name__ == '__main__':
  import argparse, pathlib
  parser = argparse.ArgumentParser()
  parser.add_argument('wsi_fn', help='the filename of target WSI')
  parser.add_argument('--dst_fn', help='save tif filename')
  args = parser.parse_args()
  if not args.dst_fn:
    args.dst_fn = '{}_tissue.tif'.format(pathlib.Path(args.wsi_fn).name[:-4])

  if not os.path.isfile(args.wsi_fn):
    print('wsi not found:', args.wsi_fn)
    exit()

  print('loading {}..'.format(args.wsi_fn))
  img = vips2numpy(pyvips.Image.new_from_file(args.wsi_fn))[:, :, :3]
  print('generating tissue mask..')
  tissue = (find_foreground(img) * 255).astype(np.uint8)
  print('saving tissue mask img..')
  tifffile.imsave(args.dst_fn, tissue, compress=9)
  print('done!')
