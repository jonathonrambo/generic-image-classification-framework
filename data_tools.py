
# --- imports --- #

from random import sample, shuffle

from os import scandir, path, makedirs
from shutil import rmtree, copy
from glob import glob

from itertools import islice

from copy import deepcopy

import re



# --- functions --- #

def regex_replace(pattern, string, replacement='', function='extract'):

    if function == 'extract':
        match = re.search(pattern, string)
        return match.group(1)
    else:
        return re.sub(pattern, replacement, string)


def create_train_test_val(img_dir, split_dir, weights=None, dir_types=('train', 'test', 'val'), n=None, total_files=10000000, rebuild=True, custom=False, custom_n=None):

    if rebuild:
        if path.exists(split_dir):
            rmtree(split_dir)

    classes = [f.name for f in scandir(img_dir) if f.is_dir() and f.name != 'dining room']
    images = glob(f'{img_dir}*/*')
    shuffle(images)

    if not custom:
        for dir_type in dir_types:
            for img_class in classes:
                path_string = f'{split_dir}/{dir_type}/{img_class}'
                if not path.exists(path_string):
                    makedirs(path_string)
    else:
        for img_class in classes:
            path_string = f'{split_dir}/{img_class}'
            if not path.exists(path_string):
                makedirs(path_string)

    for img_class in classes:
        tmp_images = [j for j in images if img_class in j]
        num_files = min(total_files, len(tmp_images))


        if not custom:
            tmp_weights = deepcopy(n) if n else [int((i * num_files) // 1) for i in weights]
            splits = [list(islice(tmp_images, i)) for i in tmp_weights]

            for i, dir_type in enumerate(dir_types):
                [copy(j, f'{split_dir}{dir_type}/{img_class}/' + regex_replace(r'\\([^/\\]+)$', j)) for j in splits[i]]
        else:
            [copy(j, f'{split_dir}/{img_class}/' + regex_replace(r'\\([^/\\]+)$', j)) for j in sample(tmp_images, custom_n)]



