import numpy as np
from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from utils.setup_elements import class_order as class_order_table


class CIFAR100(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'cifar100'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(CIFAR100, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)


    def download_load(self):
        dataset_train = datasets.CIFAR100(root=self.root, train=True, download=True)
        self.train_data = dataset_train.data
        self.train_label = np.array(dataset_train.targets)
        dataset_test = datasets.CIFAR100(root=self.root, train=False, download=True)
        self.test_data = dataset_test.data
        self.test_label = np.array(dataset_test.targets)

    def setup(self):
        if self.scenario == 'nc':
            print('nc, lt, ltio: ', self.params.lt, self.params.ltio)
            if self.params.lt: 
                class_order_var = 'cifar100_lt'
            elif self.params.ltio:
                class_order_var = 'cifar100_ltio'
            else:
                class_order_var = 'cifar100_conv'
            self.task_labels, self.data = create_task_composition(class_nums=100, num_tasks=self.task_nums, nc_first_task=self.params.nc_first_task, class_order = class_order_table[class_order_var], \
                                                                    x=self.train_data, y=self.train_label, x_test=self.test_data, y_test=self.test_label, lt=self.params.lt, ltio=self.params.ltio, fixed_order=self.params.fix_order, imb_factor=self.params.imb_factor, dataset_flag='cifar100')
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

