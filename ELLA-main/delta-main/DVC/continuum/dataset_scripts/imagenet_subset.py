import numpy as np
from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns
from utils.setup_elements import class_order as class_order_table
from collections import Counter

import os
import json
import torch
import cv2

from torch.utils.data import Dataset
import json
import random
from PIL import Image

class ImageNet_Subset(DatasetBase):
	def _get_img_from_paths(self, text_file):
		img_data = []
		data = []
		labels = []

		with open(text_file,'rb') as f:
			for line in f:
				temp = line.strip().decode("utf-8")
				temp = temp.split('/')
				end_part = temp[-1:]
				temp = '/home/raghav12/ImageNet2012/'+temp[2]+"/"+temp[3]+"/"+str(end_part[0].split(' ')[0])
				data.append(temp)
				labels.append(int(end_part[0].split(' ')[1]))


		for datapath in data:
			img = cv2.imread(datapath)
			img_data.append(img)
		img_data = np.array(img_data)
		return img_data, labels

	def __init__(self, scenario, params):
		dataset = 'imagenet_subset'
		num_tasks = params.num_tasks
		self.train_file = '/home/raghav12/ImageNet2012/train_1993_100.txt'
		self.test_file = '/home/raghav12/ImageNet2012/test_1993_100.txt'
		super(ImageNet_Subset, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)

	def download_load(self):
		self.train_data, self.train_label = self._get_img_from_paths(self.train_file)
		self.test_data, self.test_label = self._get_img_from_paths(self.test_file)

	def setup(self):
		if self.scenario == 'ni':
			self.train_set, self.val_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data, self.test_label,
                                                                                        self.task_nums, 32,
                                                                                        self.params.val_size,
                                                                                        self.params.ns_type, self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)
		elif self.scenario == 'nc':
			#vfn does not have conventional setting - class is traditionally longtailed
			if self.params.lt: 
				class_order_var = 'imagenet_subset_lt'
			elif self.params.ltio:
				class_order_var = 'imagenet_subset_ltio'
			else:
				class_order_var = 'imagenet_subset_conv'

			self.task_labels, self.data = create_task_composition(class_nums=100, num_tasks=self.task_nums, nc_first_task=self.params.nc_first_task, class_order = class_order_table[class_order_var], \
                                                                    x=self.train_data, y=self.train_label, x_test=self.test_data, y_test=self.test_label, lt=self.params.lt, ltio=self.params.ltio, fixed_order=self.params.fix_order, imb_factor=self.params.imb_factor, lfh=self.params.lfh, dataset_flag='imagenet_subset')
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
		return x_train, y_train, labels

	def new_run(self, **kwargs):
		self.setup()
		return self.test_set

	def test_plot(self):
		test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type, self.params.ns_factor)
        


