import numpy as np
from torchvision import datasets
from continuum.data_utils import create_task_composition, create_task_composition_vfn, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from utils.setup_elements import class_order as class_order_table
from collections import Counter
from utils.setup_elements import class_distribution_table_vfn

import os
import json
import torch
import cv2

from torch.utils.data import Dataset
import json
import random
from PIL import Image

class VFN(DatasetBase):
    # # def _get_img_from_paths(self, text_file):
    # # 	img_data = []
    # # 	data = []
    # # 	labels = []

    # # 	with open(text_file,'rb') as f:
    # # 		for line in f:
    # # 			temp = line.strip().decode("utf-8")
    # # 			temp = '../../VFN/'+temp[2:]
    # # 			data.append(temp.split('==')[0])
    # # 			labels.append(temp.split('==')[1])

    # # 	for datapath in data:
    # # 		img = cv2.imread(datapath)
    # # 		img_data.append(img)
    # # 	img_data = np.array(img_data)
    
    # # 	return img_data, labels
    def calculate_bbox_area(self, x1, y1, x2, y2):
        """Tính toán diện tích của bounding box."""
        # Đảm bảo chiều rộng và chiều cao không âm
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return width * height

    def resize_image(self, image, image_filename, original_label, image_annotations, target_size=224):
        if image is not None:
            # Lấy bounding box và cắt ảnh
            x1, y1, x2, y2 = image_annotations[(image_filename, original_label)]
            cropped_img = image[y1:y2, x1:x2]

            h_original, w_original = cropped_img.shape[:2]

            # Tính toán tỷ lệ co giãn để ảnh vừa khớp với target_size mà không bị biến dạng
            scale = min(target_size / h_original, target_size / w_original)

            # Tính kích thước mới sau khi co giãn
            new_w = int(w_original * scale)
            new_h = int(h_original * scale)

            # --- Lựa chọn phương pháp nội suy tối ưu cho chất lượng ảnh ---
            if new_w < w_original or new_h < h_original:
                # Nếu đang thu nhỏ ảnh (kích thước mới nhỏ hơn kích thước gốc)
                interpolation_method = cv2.INTER_AREA
            else:
                # Nếu đang phóng to ảnh (kích thước mới lớn hơn kích thước gốc)
                # Hoặc kích thước không đổi (bằng nhau)
                interpolation_method = cv2.INTER_CUBIC # INTER_CUBIC cho chất lượng phóng to tốt nhất

            # Co giãn ảnh với phương pháp nội suy đã chọn
            resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=interpolation_method)

            # Bước 3: Thêm đệm để đạt kích thước 224x224
            delta_h = target_size - new_h
            delta_w = target_size - new_w

            top = delta_h // 2
            bottom = delta_h - top
            left = delta_w // 2
            right = delta_w - left

            processed_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])

            return processed_img
        return None

    def _get_img_from_paths(self, text_file):
        
        __annotations__text_file_path = '/kaggle/working/ELLA/ELLA-main/data/annotations.txt'
        image_annotations = {}
        try:
            with open(__annotations__text_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        image_filename = parts[0]
                        try:
                            # Lưu ý: Theo code trước của bạn, thứ tự đọc là y1, x1, y2, x2
                            # Hãy đảm bảo rằng thứ tự này khớp với ý định của bạn
                            y1_float = float(parts[1])
                            x1_float = float(parts[2])
                            y2_float = float(parts[3])
                            x2_float = float(parts[4])
                            label = int(parts[5])
                            y1 = int(y1_float)
                            x1 = int(x1_float) 
                            y2 = int(y2_float)
                            x2 = int(x2_float)
                            # Tạo khóa là một tuple (tên file, label)
                            key = (image_filename, label)
                            # Giá trị là một tuple các tọa độ
                            # if image_filename == '767451.jpg':
                            #     print(line)
                            #     print(f"Processing image: {image_filename}, Coordinates: ({x1}, {y1}, {x2}, {y2})")
                            value = (x1, y1, x2, y2)
                            current_area = self.calculate_bbox_area(x1, y1, x2, y2)
                            if key in image_annotations:
                                # Nếu đã có khóa này, so sánh diện tích và giữ lại tọa độ có diện tích lớn hơn
                                existing_x1, existing_y1, existing_x2, existing_y2 = image_annotations[key]
                                existing_area = self.calculate_bbox_area(existing_x1, existing_y1, existing_x2, existing_y2)
                                if current_area > existing_area:
                                    image_annotations[key] = value
                            else:
                                image_annotations[key] = value
                        except ValueError:
                            print(f"Skipping line due to invalid coordinate or label format: {line.strip()}")
                    else:
                        print(f"Skipping malformed line: {line.strip()}")
        except FileNotFoundError:
            print(f"Error: The file {__annotations__text_file_path} was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")



        # Đảm bảo self.label_mapping và self.next_int_id được khởi tạo một lần
        # hoặc được reset nếu cần xử lý nhiều file theo cách độc lập.
        # Nếu muốn ánh xạ liên tục trên nhiều file, giữ chúng không reset.
        # Nếu mỗi lần gọi _get_img_from_paths là độc lập, hãy reset ở đây:
        label_mapping = {}
        next_int_id = 0
        mapped_labels = []
        img_data = []
        if not os.path.exists(text_file):
            print(f"Lỗi: File '{text_file}' không tồn tại.")
            return np.array([]), []

        print(f"Đang đọc dữ liệu từ: {text_file}")
        with open(text_file, 'r') as f: # Mở file ở chế độ đọc văn bản ('r')
            for line in f:
                stripped_line = line.strip()
                parts = stripped_line.split('==')

                if len(parts) == 2:
                    image_filename = parts[0]
                    original_label_str = parts[1] # Nhãn gốc là string (ví dụ: "0", "1", "3")

                    # Bước 1: Ánh xạ nhãn string sang số nguyên liên tiếp
                    if original_label_str not in label_mapping:
                        label_mapping[original_label_str] = next_int_id
                        next_int_id += 1

                    current_mapped_label = label_mapping[original_label_str]
                    mapped_labels.append(current_mapped_label)

                    # Bước 2: Xây dựng đường dẫn đầy đủ đến ảnh
                    # Đường dẫn ảnh đúng sẽ là BASE_FOLDER_PATH/Images/LABEL_FOLDER/image_filename.jpg
                    full_image_path = os.path.join(self.base_folder_path, original_label_str, image_filename)
                    # Bước 3: Đọc ảnh
                    img = cv2.imread(full_image_path)
                    img = self.resize_image(img, image_filename, int(original_label_str), image_annotations)
                    if img is not None:
                        img_data.append(img)
                    else:
                        print(f"Cảnh báo: Không thể tải ảnh từ '{full_image_path}'. Trong phần tải ảnh test vfn.")
                else:
                    print(f"Cảnh báo: Dòng không đúng định dạng trong '{text_file}': {stripped_line}")
        
        # Chuyển đổi danh sách ảnh thành mảng NumPy
        if img_data:
            img_data = np.array(img_data)
        else:
            img_data = np.array([]) # Trả về mảng rỗng nếu không có ảnh nào được tải

        print(f"Successfully download data from {text_file}. Total images: {len(img_data)}")
        #print(f"Final label mapping: {label_mapping}")

        return img_data, mapped_labels, image_annotations

    def __init__(self, scenario, params):
        
        # Initialize the base folder path of the dataset
        # you should change this path to the location where you have stored the dataset
        #self.base_folder_path = '/content/vfn_1_0/vfn_1_0/Images'
        self.base_folder_path = '/kaggle/input/vfn82-foodimages/vfn_1_0/vfn_1_0/Images'
        dataset = 'vfn'
        num_tasks = params.num_tasks
        # self.train_file = '/content/ELLA/ELLA-main/data/vfn_longtailed_train.txt'
        # self.test_file = '/content/ELLA/ELLA-main/data/vfn_longtailed_test.txt'
        self.train_file = '/kaggle/working/ELLA/ELLA-main/data/vfn_longtailed_train.txt'
        self.test_file = '/kaggle/working/ELLA/ELLA-main/data/vfn_longtailed_test.txt'
        nc_first_task = params.nc_first_task
        super(VFN, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)

    def download_load(self):
        # because vfn is not a torchvision dataset and vfn is 224x224 pixels (so heavy), load very slowly
        # we dont load all the images at once, like cifar100 or imagenet then dive into the fixed distribution
        # instead, we load the images following the fixed distribution 
        # training data set will be loaded at create_task_composition_vfn()

        #self.train_data, self.train_label = self._get_img_from_paths(self.train_file)
        self.test_data, self.test_label, self.image_annotations = self._get_img_from_paths(self.test_file)

    def setup(self):
        if self.scenario == 'nc':
            #vfn does not have conventional setting - class is traditionally longtailed
            if self.params.lt: 
                class_order_var = 'vfn_lt'
            else:
                class_order_var = 'vfn_ltio'
            self.task_labels, self.data = create_task_composition_vfn(image_annotations=self.image_annotations, class_nums=74, num_tasks=self.task_nums, nc_first_task=self.params.nc_first_task, class_order = class_order_table[class_order_var], \
                                                                    training_file_path=self.train_file, base_folder_image_path=self.base_folder_path, x_test=self.test_data, y_test=self.test_label, lt=self.params.lt, ltio=self.params.ltio, fixed_order=self.params.fix_order, imb_factor=self.params.imb_factor)
            self.test_set = []
            for labels in range(len(self.task_labels)):
                x_test, y_test = np.asarray(self.data[labels]['tst']['x']), np.asarray(self.data[labels]['tst']['y'])
                self.test_set.append((x_test, y_test))
        else:
            raise Exception('wrong scenario')

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            labels = set(y_train)
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = np.asarray(self.data[cur_task]['trn']['x']), np.asarray(self.data[cur_task]['trn']['y'])
        return x_train, y_train, labels, cur_task

    def new_run(self, **kwargs):
        self.setup()   
        return self.test_set

    def test_plot(self):
        return 0
