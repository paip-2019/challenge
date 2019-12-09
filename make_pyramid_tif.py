import skimage.io as io
import numpy as np
import pyvips

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

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

# numpy array to vips image
def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi

# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def save_pyramid_tif(save_fn, img, Q=70):
  img = numpy2vips(img)
  img.tiffsave(save_fn, tile=True, compression='jpeg', Q=Q, bigtiff=True, pyramid=True)

if __name__ == '__main__':
  import argparse, pathlib
  parser = argparse.ArgumentParser()
  parser.add_argument('wsi_fn', help='target image filename')
  parser.add_argument('--dst_fn', help='save filename')
  args = parser.parse_args()
  if not args.dst_fn:
    args.dst_fn = '{}_pyramid.tif'.format(pathlib.Path(args.wsi_fn).name[:-4])

  try:
    print('loading {}..'.format(args.wsi_fn))
    img = vips2numpy(pyvips.Image.new_from_file(args.wsi_fn))[:, :, :3]
  except:
    try:
      img = io.imread(args.wsi_fn)[:, :, 3]
    except:
      print('cannot open {}'.format(args.wsi_fn))
      exit()
  print('saving..')
  save_pyramid_tif(args.dst_fn, img)
