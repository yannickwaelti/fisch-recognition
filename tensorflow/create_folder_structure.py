import os
import re
import shutil


def main():
    in_dir = "dataset/images/raw_images"
    out_dir = "dataset/images/by_class"
    class_name = re.compile("(.*)_\d{1,5}((.jpg)|(.jpeg)|(.png))")
    created_dirs = []

    for filename in os.listdir(in_dir):
        dir_name = class_name.match(filename).group(1)
        if dir_name not in created_dirs:
            os.makedirs(out_dir + "/" + dir_name)
            created_dirs.append(dir_name)
        shutil.copyfile(in_dir + "/" + filename, out_dir + "/" + dir_name + "/" + filename)


if __name__ == "__main__":
    main()