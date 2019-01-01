"""
https://searchcode.com/codesearch/view/40141634/

read ZVI (Zeiss) image file

- incomplete support
- open uncompressed image from multi item image (Count>0)

- require OleFileIO_PL - a Python module to read MS OLE2 files
  http: //www.decalage.info/en/python/olefileio#attachments

.. code-block:: python

>>> import os, sys
>>> sys.path += [os.path.abspath(os.path.join('..', '..'))]
>>> import imsegm.utilities.data_io as tl_io
>>> path_file = os.path.join('data_images', 'others', 'sample.zvi')
>>> path_file = tl_io.update_path(path_file)
>>> n = get_layer_count(path_file)
>>> get_dir(path_file) # doctest: +ELLIPSIS
[...]
>>> for p in range(n):
...     zvi = zvi_read(path_file, p)
...     arr = zvi.Image.Array
...     arr.shape
(488, 648)
(488, 648)
(488, 648)
(488, 648)
>>> img = load_image(path_file)
>>> img.shape
(4, 488, 648)

"""

import struct
from collections import namedtuple

import OleFileIO_PL
import numpy as np


def i32(data):
    """ return int32 from len4 string"""
    low, high = struct.unpack('<hh', data[:4])
    return (high << 16) + low


def get_hex(data, n=16):
    return '|'.join(['%02x' % (ord(data[i])) for i in range(n)])


def read_struct(data, t):
    """ read a t type from data(str)"""
#    vartype = (ord(data[0]), ord(data[1]))
#    print t, vartype
    next_data = data[2:]  # skip vartype I16

    if t is '?':
        return [None, next_data]
    if t is 'EMPTY':
        return [None, next_data]
    if t is 'NULL':
        return [None, next_data]
    if t is 'I2':
        low = struct.unpack('<h', next_data[:2])
        return [low[0], next_data[2:]]
    if t is 'I4':
        r = i32(next_data[:4])
        return [r, next_data[4:]]
    if t is 'BLOB':
        size = i32(next_data[:4])
        r = next_data[4:4 + size]
        return [r, next_data[4 + size:]]
    if t is 'BSTR':
        # ! 4 extra bytes escaped
        low, high = struct.unpack('<hh', next_data[:4])
        size = (high << 16) + low
        if size > 0:
            s = struct.unpack('s', next_data[4:4 + size])
            next_data = next_data[4 + 4 + size:]
        else:
            s = ''
            next_data = next_data[4 + 4:]
        return [s, next_data]
    raise ValueError('unknown type:%s' % type)


ZviImageTuple = namedtuple(
    'ZviImageTuple',
    'Version FileName Width Height Depth PIXEL_FORMAT Count '
    'ValidBitsPerPixel m_PluginCLSID Others Layers Scaling'
)


def read_image_container_content(stream):
    """ returns a ZviImageTuple from a stream"""
    data = stream.read()
    next_data = data
    [version, next_data] = read_struct(next_data, 'I4')
#    [Type, next] = read_struct(next, 'I4')
#    [TypeDescription, next] = read_struct(next, 'BSTR')
    [filename, next_data] = read_struct(next_data, 'BSTR')
    [width, next_data] = read_struct(next_data, 'I4')
    [height, next_data] = read_struct(next_data, 'I4')
    [depth, next_data] = read_struct(next_data, 'I4')
    [pixel_format, next_data] = read_struct(next_data, 'I4')
    [count, next_data] = read_struct(next_data, 'I4')
    [valid_bits_per_pixel, next_data] = read_struct(next_data, 'I4')
    [m_PluginCLSID, next_data] = read_struct(next_data, 'I4')
    [others, next_data] = read_struct(next_data, 'I4')
    [layers, next_data] = read_struct(next_data, 'I4')
    [scaling, _] = read_struct(next_data, 'I2')

    zvi_image = ZviImageTuple(version, filename, width, height, depth,
                              pixel_format, count, valid_bits_per_pixel,
                              m_PluginCLSID, others, layers, scaling)
    return zvi_image


ZviItemTuple = namedtuple(
    'ZviItemTuple',
    'Version FileName Width Height Depth PIXEL_FORMAT Count '
    'ValidBitsPerPixel Others Layers Scaling Image'
)


PIXEL_FORMAT = {
    1: (3, 'ByteBGR'),
    2: (4, 'ByteBGRA'),
    3: (1, 'Byte'),
    4: (2, 'Word'),
    5: (4, 'Long'),
    6: (4, 'Float'),
    7: (8, 'Double'),
    8: (6, 'WordBGR'),
    9: (4, 'LongBGR'),
}


def read_item_storage_content(stream):
    """ returns ZviItemTuple from the stream"""
    data = stream.read()
    next_data = data
    [version, next_data] = read_struct(next_data, 'I4')
#    [Type, next] = read_struct(next, 'I4')
#    [TypeDescription, next] = read_struct(next, 'BSTR')
    [filename, next_data] = read_struct(next_data, 'BSTR')
    [width, next_data] = read_struct(next_data, 'I4')
    [height, next_data] = read_struct(next_data, 'I4')
    [depth, next_data] = read_struct(next_data, 'I4')
    [pixel_format, next_data] = read_struct(next_data, 'I4')
    [count, next_data] = read_struct(next_data, 'I4')
    [valid_bits_per_pixel, next_data] = read_struct(next_data, 'I4')
    [others, next_data] = read_struct(next_data, 'BLOB')
    [layers, next_data] = read_struct(next_data, 'BLOB')
    [scaling, _] = read_struct(next_data, 'BLOB')
    # offset is image size + header size(28)
    offset = width * height * PIXEL_FORMAT[pixel_format][0] + 28
    # parse the actual image data
    image = parse_image(data[-offset:])
    # group results into one single structure (namedtuple)
    zvi_item = ZviItemTuple(version, filename, width, height, depth,
                            pixel_format, count, valid_bits_per_pixel, others,
                            layers, scaling, image)
    return zvi_item


ImageTuple = namedtuple(
    'ImageTuple',
    'Version Width Height Depth PixelWidth PIXEL_FORMAT '
    'ValidBitsPerPixel Array'
)


def parse_image(data):
    """ returns ImageTuple from raw image data(header+image)"""
    version = i32(data[:4])
    width = i32(data[4:8])
    height = i32(data[8:12])
    depth = i32(data[12:16])
    pixel_width = i32(data[16:20])
    pixel_format = i32(data[20:24])
    valid_bits_per_pixel = i32(data[24:28])
    raw = np.fromstring(data[28:], 'uint16')
    array = np.reshape(raw, (height, width))
    image = ImageTuple(version, width, height, depth, pixel_width,
                       pixel_format, valid_bits_per_pixel, array)
    return image


def get_layer_count(file_name, ole=None):
    """ returns the number of image planes"""
    if ole is None:
        ole = OleFileIO_PL.OleFileIO(file_name)
    s = ['Image', 'Contents']
    stream = ole.openstream(s)
    zvi_image = read_image_container_content(stream)
    return zvi_image.Count


def get_dir(file_name, ole=None):
    """ returns the content structure(streams) of the zvi file
    + length of each streams """
    dirs = []
    if ole is None:
        ole = OleFileIO_PL.OleFileIO(file_name)
    for s in ole.listdir():
        stream = ole.openstream(s)
        dirs.append('%10d %s' % (len(stream.read()), s))
    return dirs


def zvi_read(fname, plane, ole=None):
    """ returns ZviItemTuple of the plane from zvi file fname """
    if ole is None:
        ole = OleFileIO_PL.OleFileIO(fname)
    s = ['Image', 'Item(%d)' % plane, 'Contents']
    stream = ole.openstream(s)
    return read_item_storage_content(stream)


def load_image(path_img):
    ole = OleFileIO_PL.OleFileIO(path_img)
    nb = get_layer_count('', ole=ole)
    # logging.debug('Count layers = %i', nb)
    image = []
    for i in range(nb):
        zvi = zvi_read('', i, ole=ole)
        image.append(zvi.Image.Array)
    image = np.array(image)
    return image
