import os
import shutil

side_map = {"2": 2, "3": 3, "l": 2, "r": 3}


def get_image_path(folder, frame_index, side):
    f_str = "{:010d}{}".format(frame_index, '.jpg')
    image_path = os.path.join(
        '/SSD/Kitti/', folder, "image_0{}/data".format(side_map[side]), f_str)
    return image_path


with open('train_files.txt', 'r') as f:
    files = f.readlines()
    files = [i.strip() for i in files]

new_files = files[:1000]

with open('subset.txt', 'w') as f:
    for i in new_files:
        print(i, file=f)

for i in new_files:
    folder, frame_index, side = i.split()

    for f_id in [0, -1, 1]:
        img_path = get_image_path(folder, f_id + int(frame_index), side)
        save_path = img_path.replace('/SSD/Kitti/', '../images/')
        dir, name = os.path.split(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(img_path, save_path)

