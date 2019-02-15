from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import sys
import shutil
import urllib.request
import tarfile
import zipfile
import progressbar
from subprocess import call
from tqdm import tqdm
from os.path import join as pjoin

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0

    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_data(dataset, raw_dir):
    create_dirs([raw_dir])

    if dataset == "CUB200":
        urls = ["http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"]
    elif dataset == "MIT67":
        urls = ["http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar",
                "http://web.mit.edu/torralba/www/TrainImages.txt",
                "http://web.mit.edu/torralba/www/TestImages.txt"]
    elif dataset == "TINY_IMAGENET":
        urls = ["http://cs231n.stanford.edu/tiny-imagenet-200.zip"]

    for i, url in enumerate(urls):
        filename = pjoin(raw_dir, url.split('/')[-1])

        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename, show_progress)

            print("{} sucessfully downloaded".format(dataset))
        else:
            print("{} already downloaded".format(dataset))

        if filename.endswith("tgz") or filename.endswith("tar"):
            tar_ref = tarfile.open(filename)
            tar_ref.extractall(raw_dir)
            tar_ref.close()
        elif filename.endswith("zip"):
            zip_ref = zipfile.ZipFile(filename, 'r')
            zip_ref.extractall(raw_dir)
            zip_ref.close()

def process_data(dataset, data_dir, raw_dir):
    create_dirs([data_dir])

    if dataset == "CUB200":
        # These path should be fine
        images_txt_path = pjoin(raw_dir, "CUB_200_2011", 'images.txt')
        train_test_split_path = pjoin(raw_dir, "CUB_200_2011", 'train_test_split.txt')
        images_path = pjoin(raw_dir, "CUB_200_2011", "images")

        # Here declare where you want to place the train/test folders
        # You don't need to create them!
        test_folder = pjoin(data_dir, "test")
        train_folder = pjoin(data_dir, "train")

        def ignore_files(data_dir,files): return [f for f in files if os.path.isfile(os.path.join(data_dir,f))]

        shutil.copytree(images_path,test_folder,ignore=ignore_files)
        shutil.copytree(images_path,train_folder,ignore=ignore_files)

        with open(images_txt_path) as f:
          images_lines = f.readlines()

        with open(train_test_split_path) as f:
          split_lines = f.readlines()

        test_images, train_images = 0,0

        for image_line, split_line in tqdm(zip(images_lines,split_lines)):

          image_line = (image_line.strip()).split(' ')
          split_line = (split_line.strip()).split(' ')

          image = plt.imread(pjoin(images_path, image_line[1]))

          # Use only RGB images, avoid grayscale
          if len(image.shape) == 3:

            # If test image
            if(int(split_line[1]) is 0):
              shutil.copyfile(pjoin(images_path, image_line[1]), pjoin(test_folder, image_line[1]))
              test_images += 1
            else:
              # If train image
              shutil.copyfile(pjoin(images_path,image_line[1]), pjoin(train_folder, image_line[1]))
              train_images += 1

        print(train_images, test_images)
        assert train_images == 5990
        assert test_images == 5790

        print('Dataset succesfully splitted!')
        shutil.rmtree(raw_dir)

    elif dataset == "MIT67":
        train_txt = pjoin(raw_dir, "TrainImages.txt")
        test_txt = pjoin(raw_dir, "TestImages.txt")
        images_path = pjoin(raw_dir, "Images")

        with open(train_txt) as f:
            train_list = f.read().splitlines()
        with open(test_txt) as f:
            test_list = f.read().splitlines()

        print(train_list)
        print(test_list)

        train_folder = pjoin(data_dir, "train")
        test_folder = pjoin(data_dir, "test")

        def ignore_files(data_dir, files): return [
            f for f in files if os.path.isfile(os.path.join(data_dir,f))]

        shutil.copytree(images_path, test_folder, ignore=ignore_files)
        shutil.copytree(images_path, train_folder, ignore=ignore_files)

        train_images, test_images = 0, 0

        for folder, _, image_files in os.walk(images_path):
            last_folder = folder.split("/")[-1]
            for image_file in image_files:
                name = pjoin(last_folder, image_file)
                if name in train_list:
                    shutil.copyfile(pjoin(images_path, name), pjoin(train_folder, name))
                    train_images += 1
                elif name in test_list:
                    shutil.copyfile(pjoin(images_path, name), pjoin(test_folder, name))
                    test_images += 1

        print(train_images, test_images)
        assert train_images == 5360
        assert test_images == 1340

        print('Dataset succesfully splitted!')
        shutil.rmtree(raw_dir)

    elif dataset == "TINY_IMAGENET":
        for mode in ["train", "test", "val"]:
            shutil.move(pjoin(raw_dir, "tiny-imagenet-200", mode), data_dir)

        shutil.rmtree(raw_dir)

def download_and_process_data(dataset, base_dir):
    raw_dir = pjoin(base_dir, dataset, "raw")
    data_dir = pjoin(base_dir, dataset)
    if not os.path.exists(data_dir):
        download_data(dataset, raw_dir)
        process_data(dataset, data_dir, raw_dir)
    else:
        print("Data already existing in {}".format(data_dir))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", nargs='+', type=str)
    args = parser.parse_args()
    base_dir = "./data"
    for dataset in args.datasets:
        raw_dir = pjoin(base_dir, dataset, "raw")
        data_dir = pjoin(base_dir, dataset)
        download_data(dataset, raw_dir)
        process_data(dataset, data_dir, raw_dir)
