import os
import re
import shutil


class_name = re.compile(r"(.*)_\d{1,5}((.jpg)|(.jpeg)|(.png))")


def main():
    in_dir = "dataset/images/raw_images"
    out_dir = "dataset/images/by_class"
    out_file = "dataset/labels.txt"
    class_list = split_into_folders(in_dir, out_dir)
    write_class_list(class_list, out_file)


def split_into_folders(in_dir, out_dir):
    class_list = []

    for filename in os.listdir(in_dir):
        dir_name = class_name.match(filename).group(1)
        if dir_name not in class_list:
            class_list.append(dir_name)
            try:
                os.makedirs(out_dir + "/" + dir_name)


                shutil.copyfile(in_dir + "/" + filename, out_dir + "/" + dir_name + "/" + filename)
            except FileExistsError:
                pass
    return class_list


def write_class_list(class_list, out_file):
    with open(out_file, "w") as f:
        f.write("\n".join(class_list))

if __name__ == "__main__":
    main()
