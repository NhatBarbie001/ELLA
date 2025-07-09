#python imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pickle import FALSE
import random
from PIL import Image
#torch imports
from torch.utils.data import Dataset
from torch.utils import data
#specific code imports 
from utils.setup_elements import transforms_match, class_distribution_table, class_distribution_table_imagenet, class_distribution_table_vfn
from collections import Counter

# def create_task_composition(class_nums, num_tasks, fixed_order=False):
#     classes_per_task = class_nums // num_tasks
#     total_classes = classes_per_task * num_tasks
#     label_array = np.arange(0, total_classes)
#     if not fixed_order:
#         np.random.shuffle(label_array)

#     task_labels = []
#     for tt in range(num_tasks):
#         tt_offset = tt * classes_per_task
#         task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
#         print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
#     return task_labels

def create_task_composition(class_nums, num_tasks, nc_first_task, class_order, \
                            x, y, x_test, y_test, lt=False, ltio=False, fixed_order=True, imb_factor=0.01, dataset_flag = 'imagenet_subset'):
    
    ####create order of classes and split tasks with certain class order####
    task_labels = []
    clsanalysis={}
    data = {}
    count = 0
    print('class nums: ', class_nums)
    train_dist = []
    test_dist = []
    
    if class_order is None:
        class_order = list(range(class_nums))
    else:
        class_nums = len(class_order)
        class_order = class_order.copy()
    if fixed_order is False: #shuffle is True
        np.random.shuffle(class_order)
        
    if nc_first_task is None:
        cpertask = np.array([class_nums // num_tasks] * num_tasks)
        for i in range(class_nums % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < class_nums, "first task wants more classes than available"
        remaining_classes = class_nums - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class available per task"
        cpertask = np.array([nc_first_task]+[remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1
    
    assert class_nums == cpertask.sum(), "something is wrong, split doesnt add up"
    # print('cpertask now: ', cpertask.tolist())
    # print('class_order', class_order)
    
    tt_offset = 0
    for tt in range(num_tasks):
#         tt_offset += tt * cpertask[tt]
        task_labels.append(class_order[tt_offset:tt_offset + cpertask[tt]])
        tt_offset += cpertask[tt]
        print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
        
    ####create the train and test split based on the order of classes created above####
    assert class_nums == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))
    # print('init class: ', init_class)
    #initalize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        clsanalysis[tt] = np.zeros(cpertask[tt])
    #TRAIN ANALYSIS
    num_per_cls = np.zeros(class_nums)
    for i in range(len(x)):
        this_image = x[i]
        this_label = y[i]
        if this_label not in class_order:
            continue
        this_label = class_order.index(this_label) 
        this_task = (this_label >= cpertask_cumsum).sum()
        num_per_cls[this_label] += 1
    if dataset_flag == 'cifar100':
        
        if lt:
            print('lt?')
            dist = 'exp'
            img_num_per_cls = class_distribution_table['lt']
        elif ltio:
            print('ltio?')
            img_num_per_cls = class_distribution_table['ltio']
        else:
            print('cifar, lt: ', lt, ' ltio: ', ltio)
            dist = 'conv'
            img_num_per_cls = class_distribution_table['conv']
    elif dataset_flag == 'imagenet_subset':
        if lt:
            dist = 'exp'
            img_num_per_cls = class_distribution_table_imagenet['lt']
        elif ltio:
            img_num_per_cls = class_distribution_table_imagenet['ltio']
        else:
            dist = 'conv'
            img_num_per_cls = class_distribution_table_imagenet['conv']

    # print('class order: ', class_order)
    # print('img per cls: ', img_num_per_cls)
    num_per_cls_now = np.zeros(class_nums)
    #ALL or Train
    for i in range(len(x)):
        this_image = x[i]
        this_label = y[i]
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label_old = this_label
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if num_per_cls_now[this_label] >= img_num_per_cls[this_label] and (ltio or lt):
            continue
        else:
            clsanalysis[this_task][this_label - init_class[this_task]] += 1
            data[this_task]['trn']['x'].append(this_image)
            data[this_task]['trn']['y'].append(this_label_old) #- init_class[this_task])
            num_per_cls_now[this_label] += 1

    # print('num per cls now: ', num_per_cls_now)
    # ALL OR TEST
    for i in range(len(x_test)):
        this_image = x_test[i]
        this_label = y_test[i]
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label_old = this_label
        this_label = class_order.index(this_label) 
        
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label_old)# - init_class[this_task])

    
    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # #test our retrun op
    print('task labels: ', task_labels)
    
    for i in range(num_tasks):
        train_dist.append(len(data[i]['trn']['y']))
        # print('unique in train task: ',unique(data[i]['trn']['y']))
        # print('unique in test task: ',unique(data[i]['tst']['y']))
        # print('\n')
        test_dist.append(len(data[i]['tst']['y']))
    print('training distribution: ', train_dist)
    print('test distribtuion: ', test_dist)

    # print(len(data[0]['trn']['y']), np.sum(img_num_per_cls[:5]))
    return task_labels, data

def create_task_composition_vfn(class_nums, num_tasks, nc_first_task, class_order, \
                            training_file_path, base_folder_image_path, x_test, y_test, lt=False, ltio=False, fixed_order=True, imb_factor=0.01):

    print('nc_first_task: ', nc_first_task)
    ####create order of classes and split tasks with certain class order####
    task_labels = []
    clsanalysis={}
    data = {}
    train_dist = []
    test_dist = []
    
    if class_order is None:
        class_order = list(range(class_nums))
    else:
        class_nums = len(class_order)
        class_order = class_order.copy()
    if fixed_order is False: #shuffle is True
        np.random.shuffle(class_order)

    num_per_cls = np.zeros(class_nums)
    label_mapping = {}
    next_int_id = 0
    with open(training_file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            parts = stripped_line.split('==')
            if len(parts) == 2:
                image_filename = parts[0]
                original_label = int(parts[1])
                if original_label_str not in self.label_mapping:
                    self.label_mapping[original_label] = self.next_int_id
                    self.next_int_id += 1

                current_mapped_label = self.label_mapping[original_label]
                num_per_cls[current_mapped_label] += 1

                


    if lt:
        img_num_per_cls = class_distribution_table_vfn['lt']

        # because the VFN74 is not balanced, maybe there is a class that has less images than that in the fixed distribution
        # so we need to shuffle the class order until it is valid
        while True:
            current_order_is_valid = True
            np.random.shuffle(class_order)
            for i, class_id in enumerate(class_order):
                acctual_count = num_per_cls[class_id]
                if acctual_count < img_num_per_cls[i]:
                    current_order_is_valid = False
                    break
            if current_order_is_valid:
                break
                
    else:
        img_num_per_cls = class_distribution_table_vfn['ltio']

        
    if nc_first_task is None:
        cpertask = np.array([class_nums // num_tasks] * num_tasks)
        for i in range(class_nums % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < class_nums, "first task wants more classes than available"
        remaining_classes = class_nums - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class available per task"
        cpertask = np.array([nc_first_task]+[remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1
        print('cpertask: ', cpertask)
    assert class_nums == cpertask.sum(), "something is wrong, split doesnt add up"
    
    tt_offset = 0 
    for tt in range(num_tasks):
        task_labels.append(class_order[tt_offset:tt_offset + cpertask[tt]])
        tt_offset += cpertask[tt]
        print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
        
    ####create the train and test split based on the order of classes created above####
    assert class_nums == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))
    #initalize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        clsanalysis[tt] = np.zeros(cpertask[tt])
    #TRAIN ANALYSIS
    #num_per_cls = np.zeros(class_nums)
    # for i in range(len(x)):
    #     this_image = x[i]
    #     this_label = int(y[i])
    #     if this_label not in class_order:
    #         continue
    #     this_label = class_order.index(this_label)
    #     this_task = (this_label >= cpertask_cumsum).sum()
    #     num_per_cls[this_label] += 1
    # if lt:
    #     img_num_per_cls = class_distribution_table_vfn['lt']
    # else:
    #     img_num_per_cls = class_distribution_table_vfn['ltio']
    # img_num_per_cls = num_per_cls # as this datatset is longtailed by itself
    num_per_cls_now = np.zeros(class_nums)
    # print('class order: ', class_order)
    # print('img per cls: ', img_num_per_cls)
    #ALL or Train

    with open(training_file_path, 'r') as f:
        for line in f:
                stripped_line = line.strip()
                parts = stripped_line.split('==')

                if len(parts) == 2:
                    image_filename = parts[0]
                    original_label = int(parts[1])
                    this_label = self.label_mapping[original_label]
                    if this_label not in class_order:
                        continue
                    this_label_old = this_label
                    this_label = class_order.index(this_label)
                    this_task = (this_label >= cpertask_cumsum).sum()
                    if num_per_cls_now[this_label] >= img_num_per_cls[this_label] and (ltio or lt):
                        continue
                    else:
                        clsanalysis[this_task][this_label - init_class[this_task]] += 1
                        full_image_path = os.path.join(base_folder_image_path, str(original_label), image_filename)
                        this_image = cv2.imread(full_image_path)
                        data[this_task]['trn']['x'].append(this_image)
                        data[this_task]['trn']['y'].append(this_label_old) #- init_class[this_task])
                        num_per_cls_now[this_label] += 1




    # for i in range(len(x)):
    #     this_image = x[i]
    #     this_label = int(y[i])
    #     if this_label not in class_order:
    #         continue
    #     # If shuffling is false, it won't change the class number
    #     this_label_old = this_label
    #     this_label = class_order.index(this_label)
    #     # add it to the corresponding split
    #     this_task = (this_label >= cpertask_cumsum).sum()
    #     if num_per_cls_now[this_label] >= img_num_per_cls[this_label] and (ltio or lt):
    #         continue
    #     else:
    #         clsanalysis[this_task][this_label - init_class[this_task]] += 1
    #         data[this_task]['trn']['x'].append(this_image)
    #         data[this_task]['trn']['y'].append(this_label_old) #- init_class[this_task])
    #         num_per_cls_now[this_label] += 1
    # print('num per cls now: ', num_per_cls_now)
    # ALL OR TEST
    for i in range(len(x_test)):
        this_image = x_test[i]
        this_label = int(y_test[i])
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label_old = this_label
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label_old) #- init_class[this_task])

    for tt in range(num_tasks):
       # print('tt: ', tt, 'num_tasks: ', num_tasks)
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        #print('data[tt][cla]: ', len(np.unique(data[tt]['trn']['y'])))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # #test our retrun op
    print('task labels: ', task_labels)
    
    for i in range(num_tasks):
        train_dist.append(len(data[i]['trn']['y']))
        # print('unique in train task: ',unique(data[i]['trn']['y']))
        # print('unique in test task: ',unique(data[i]['tst']['y']))
        # print('\n')
        test_dist.append(len(data[i]['tst']['y']))
    print('training distribution: ', train_dist)
    print('test distribtuion: ', test_dist)

    return task_labels, data

def load_task_with_labels_torch(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((y == i).nonzero().view(-1))
    idx = torch.cat(tmp)
    return x[idx], y[idx]


def load_task_with_labels(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((np.where(y == i)[0]))
    idx = np.concatenate(tmp, axis=None)
    return x[idx], y[idx]



class dataset_transform(data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.transform = transform  # save the transform

    def __len__(self):
        return len(self.y)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, idx):
        # return the augmented image
        if self.transform:
            x = self.transform(self.x[idx])
        else:
            x = self.x[idx]

        return x.float(), self.y[idx]


def setup_test_loader(test_data, params):
    test_loaders = []

    for (x_test, y_test) in test_data:
        test_dataset = dataset_transform(x_test, y_test, transform=transforms_match[params.data])
        test_loader = data.DataLoader(test_dataset, batch_size=params.test_batch, shuffle=True, num_workers=0)
        test_loaders.append(test_loader)
    return test_loaders


def shuffle_data(x, y):
    perm_inds = np.arange(0, x.shape[0])
    np.random.shuffle(perm_inds)
    rdm_x = x[perm_inds]
    rdm_y = y[perm_inds]
    return rdm_x, rdm_y


def train_val_test_split_ni(train_data, train_label, test_data, test_label, task_nums, img_size, val_size=0.1):
    train_data_rdm, train_label_rdm = shuffle_data(train_data, train_label)
    val_size = int(len(train_data_rdm) * val_size)
    val_data_rdm, val_label_rdm = train_data_rdm[:val_size], train_label_rdm[:val_size]
    train_data_rdm, train_label_rdm = train_data_rdm[val_size:], train_label_rdm[val_size:]
    test_data_rdm, test_label_rdm = shuffle_data(test_data, test_label)
    train_data_rdm_split = train_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    train_label_rdm_split = train_label_rdm.reshape(task_nums, -1)
    val_data_rdm_split = val_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    val_label_rdm_split = val_label_rdm.reshape(task_nums, -1)
    test_data_rdm_split = test_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    test_label_rdm_split = test_label_rdm.reshape(task_nums, -1)
    return train_data_rdm_split, train_label_rdm_split, val_data_rdm_split, val_label_rdm_split, test_data_rdm_split, test_label_rdm_split