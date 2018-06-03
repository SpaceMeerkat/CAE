import os
from torch.utils import data
from astropy.io import fits
from skimage.transform import resize


IMG_EXTENSIONS = [
    ".fits"
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def default_fits_loader(file_name: str, img_size: tuple, slice_index):
    file = fits.open(file_name)
    _data = file[1].data
    _data = resize(_data[slice_index], img_size, mode='reflect')
    _label = file[0].header['LABEL']

    if len(_data.shape) < 3:
        _data = _data.reshape((*_data.shape, 1))
		
	# set all NaN values to zero
    _data[_data != _data] = 0

    return _data, _label


class FITSCubeDataset(data.Dataset):
    def __init__(self, data_path, cube_length, transforms, img_size):
        self.data_path = data_path
        self.transforms = transforms
        self.img_size = img_size
        self.cube_length = cube_length
        self.img_files = make_dataset(data_path)

    def __getitem__(self, index):
        cube_index = index // self.cube_length
        slice_index = index % self.cube_length
        _img, _label = default_fits_loader(self.img_files[cube_index], self.img_size, slice_index)
        if self.transforms is not None:
            _data = (self.transforms(_img), _label)
        else:
            _data = (_img, _label)

        return _data

    def __len__(self):
        return len(self.img_files) * self.cube_length
