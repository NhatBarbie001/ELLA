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
    def _get_img_from_paths(self, text_file):



        # Đảm bảo self.label_mapping và self.next_int_id được khởi tạo một lần
        # hoặc được reset nếu cần xử lý nhiều file theo cách độc lập.
        # Nếu muốn ánh xạ liên tục trên nhiều file, giữ chúng không reset.
        # Nếu mỗi lần gọi _get_img_from_paths là độc lập, hãy reset ở đây:
        self.label_mapping = {}
        self.next_int_id = 0
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
                    if original_label_str not in self.label_mapping:
                        self.label_mapping[original_label_str] = self.next_int_id
                        self.next_int_id += 1
                    
                    current_mapped_label = self.label_mapping[original_label_str]
                    mapped_labels.append(current_mapped_label)

                    # Bước 2: Xây dựng đường dẫn đầy đủ đến ảnh
                    # Đường dẫn ảnh đúng sẽ là BASE_FOLDER_PATH/Images/LABEL_FOLDER/image_filename.jpg
                    full_image_path = os.path.join(self.base_folder_path, original_label_str, image_filename)
                    
                    # Bước 3: Đọc ảnh
                    img = cv2.imread(full_image_path)
                    if img is not None:
                        img_data.append(img)
                    else:
                        print(f"Cảnh báo: Không thể tải ảnh từ '{full_image_path}'. Bỏ qua ảnh này.")
                else:
                    print(f"Cảnh báo: Dòng không đúng định dạng trong '{text_file}': {stripped_line}")
        
        # Chuyển đổi danh sách ảnh thành mảng NumPy
        if img_data:
            img_data = np.array(img_data)
        else:
            img_data = np.array([]) # Trả về mảng rỗng nếu không có ảnh nào được tải

        print(f"Hoàn tất tải ảnh từ {text_file}. Tổng số ảnh tải được: {len(img_data)}")
        print(f"Ánh xạ nhãn cuối cùng: {self.label_mapping}")
        
        return img_data, mapped_labels

    def __init__(self, scenario, params):
        
        # Initialize the base folder path of the dataset
        # you should change this path to the location where you have stored the dataset
        self.base_folder_path = '/content/drive/MyDrive/vfn_1_0/vfn_1_0/Images'
        dataset = 'vfn'
        num_tasks = params.num_tasks
        self.train_file = '/content/ELLA/ELLA-main/data/vfn_longtailed_train.txt'
        self.test_file = '/content/ELLA/ELLA-main/data/vfn_longtailed_test.txt'
        nc_first_task = params.nc_first_task
        super(VFN, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)

    def download_load(self):
        # because vfn is not a torchvision dataset and vfn is 224x224 pixels (so heavy), load very slowly
        # we dont load all the images at once, like cifar100 or imagenet then dive into fixed distribution
        # instead, we load the images following the fixed distribution 
        # training data set will be loaded at create_task_composition_vfn()

        #self.train_data, self.train_label = self._get_img_from_paths(self.train_file)
        self.test_data, self.test_label = self._get_img_from_paths(self.test_file)

    def setup(self):
        if self.scenario == 'nc':
            #vfn does not have conventional setting - class is traditionally longtailed
            if self.params.lt: 
                class_order_var = 'vfn_lt'
            else:
                class_order_var = 'vfn_ltio'
            self.task_labels, self.data = create_task_composition_vfn(class_nums=74, num_tasks=self.task_nums, nc_first_task=self.params.nc_first_task, class_order = class_order_table[class_order_var], \
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
