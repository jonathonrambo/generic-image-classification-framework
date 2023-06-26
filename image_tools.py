

# --- imports --- #

import numpy as np
from random import shuffle

from multiprocessing import Pool, cpu_count

from os import remove
from pathlib import Path
from glob import glob

import augly.image as imaugs
from PIL import Image


# --- functions --- #

def resize_images_worker(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((150, 150))
        image.save(image_path)
    except:
        remove(image_path)


def resize_images(images):
    pool = Pool(cpu_count() - 2)
    pool.map(resize_images_worker, images)
    pool.close()
    pool.join()


def augment_worker(img_path):

    image = Image.open(img_path)

    transforms = [[imaugs.Skew, {'skew_factor': 0.1}, 0.1],
                  [imaugs.RandomNoise, {'mean': -0.1, 'var': 0.02}, 0.75],
                  [imaugs.RandomBlur, {'min_radius': 0.2, 'max_radius': 0.4}, 0.4],
                  [imaugs.RandomRotation, {'min_degrees': 10, 'max_degrees': 20}, 0.4],
                  [imaugs.HFlip, {}, 0.5],
                  [imaugs.Scale, {'factor': np.random.randint(6, 9) / 10}, 0.3],
                  [imaugs.Opacity, {'level': np.random.randint(5, 9) / 10}, 0.6]]

    weights = np.array([i[2] for i in transforms])

    transform_idx = np.random.choice(list(range(len(transforms))), np.random.randint(2, 4), replace=False, p=weights / sum(weights))
    transforms = [transforms[i] for i in transform_idx]
    shuffle(transforms)

    aug = imaugs.Compose([i[0](**i[1]) for i in transforms])
    image = aug(image)
    image = image.resize((150, 150))

    iteration = 0

    while Path(f"{img_path[:img_path.rfind('.')]}(edited-{iteration}){img_path[img_path.rfind('.'):]}").is_file():
        iteration += 1

    image.save(f"{img_path[:img_path.rfind('.')]}(edited-{iteration}){img_path[img_path.rfind('.'):]}")


def augment_images(imgs):

    pool = Pool(cpu_count() - 2)
    pool.map(augment_worker, imgs)
    pool.close()
    pool.join()


def augmentation_controller(iterations, train_dir, classes, rebuild=True):
    if rebuild:
        for file in [i for i in glob(f'{train_dir}*/*') if 'edited-' in i]:
            remove(file)

    print('classes:')
    [print(f'\t{i}') for i in classes]

    class_dirs = [f'{train_dir}{train_class}/' for train_class in classes]
    image_paths = [[i for i in glob(f'{class_dir}*') if 'edited' not in i] for class_dir in class_dirs]
    print(f'\ntotal images - {len([element for sublist in image_paths for element in sublist])}\n')

    for iter_num in range(iterations):
        print(f'iteration - {iter_num + 1}')
        for idx, i in enumerate(image_paths):
            augment_images(i)


