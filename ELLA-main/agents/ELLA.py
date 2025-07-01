
import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#  import MRFA
from methods.mrfa import MRFA

def update_distribution(dist, current_data):

    for data in current_data:
        temp_label = int(data)
        dist[int(temp_label)] += 1

    # print('dist vec: ',dist)
    return dist

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, total_tasks, class_size, which_task):
        super().__init__()
        # numerator = (class_size / total_tasks) * (which_task+1)
        cls_prior = cls_num_list / sum(cls_num_list)
        cls_prior = torch.FloatTensor(cls_prior).cuda()
        self.log_prior = torch.log(cls_prior).unsqueeze(0)

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior
        # print('min and max target balanced softmax: ', labels.min(), labels.max())
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss

class ELLA(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ELLA, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters


        self.perturb_p = np.full(5, params.ELLA_beta)
        self.ELLA_alpha = params.ELLA_alpha

        self.convnet_type = 'resnet18'
        # Khởi tạo MRFA
        self.MRFA = MRFA()


        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        )
        if params.data == 'cifar100' or params.data == 'imagenet_subset':
            self.class_size = 100
        else:
            self.class_size = 74

        self.tasks = params.num_tasks
        self.lr = params.learning_rate
        self.wd = params.weight_decay

    def train_learner(self, x_train, y_train, which_task):
        self.before_train(x_train, y_train)
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)
        losses = AverageMeter()
        acc_batch = AverageMeter()


        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                distribution_vector = np.zeros(self.class_size, dtype='float')
                for j in range(self.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    for param in self.model.encoder.parameters():
                        param.requires_grad = True
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        distribution_vector = update_distribution(distribution_vector, torch.cat((batch_y, mem_y)))
                        loss_func = BalancedSoftmaxLoss(np.array(distribution_vector), self.tasks, self.class_size, which_task).cuda()

                        combined_batch = torch.cat((batch_x, mem_x))
                        combined_labels = torch.cat((batch_y, mem_y))
                        combined_batch_aug = self.transform(combined_batch)
    #=========================================================================================================
                        self.MRFA._init_inbatch_properties()
                        mem_x_aug = combined_batch_aug[batch_x.size(0):batch_x.size(0) + mem_x.size(0)]

                        perturb_mask = torch.zeros(batch_x.size(0) + mem_x.size(0), dtype=torch.bool)
                        perturb_mask[batch_x.size(0):] = True  # Đánh dấu các sample từ memory

                        self.MRFA.perturbation_idices.extend(np.arange(mem_x.size(0)).tolist())
                        self.MRFA.perturbation_idices_inbatch.extend(perturb_mask.nonzero().flatten().tolist())

                        self.MRFA.perturbation_layers.extend(np.random.randint(0, len(self.perturb_p), mem_x.size(0)).tolist())
                        
                        self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * np.random.rand(mem_x.size(0))).tolist()
                        ############################
                        # sửa thành rondom trong khoảng (0.25-0.75)
                        # self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * (0.25 + 0.5 * np.random.rand(mem_x.size(0)))).tolist()
                        # Sử dụng mixed precision
                        with torch.cuda.amp.autocast():
                            #features_combined_batch = self.model.forward(combined_batch).unsqueeze(1)

                            out_stage1 = self.model.logits(combined_batch)
                            loss_stage2_from1 = loss_func(out_stage1, combined_labels)

                            self.MRFA.feature_augmentation(
                                model=self.model,
                                convnet=self.model.encoder,  # Thêm tham số convnet
                                samples=mem_x,
                                samples_aug=mem_x_aug,
                                targets=mem_y,
                                net_type=self.convnet_type,
                                criterion=self.criterion,
                                balanced_loss_stage1 = loss_stage2_from1,
                                ELLAalpha = self.ELLA_alpha,
                                for_mem_x_aug=True
                            )
                            self.MRFA.register_perturb_forward_prehook(self.model, self.convnet_type)
                            features_combined_batch = self.model.forward(combined_batch).unsqueeze(1)
                            # ở MRFA get feature augmentation ta forward combined_batch trước
                            # Xóa 4 phần tử đầu tiên của self.MRFA.perturbations
                            # còn lại 4 phần tử cuối là input gradient của combined_batch_aug
                            self.MRFA.perturbations = self.MRFA.perturbations[4:]

                            # random lại hệ số nhiễu
                            self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * np.random.rand(mem_x.size(0))).tolist()
                            # random lại layer cho các mẫu
                            self.MRFA.perturbation_layers = []
                            self.MRFA.perturbation_layers.extend(np.random.randint(0, len(self.perturb_p), mem_x.size(0)).tolist())

                            features_combined_batch_aug = self.model.forward(combined_batch_aug).unsqueeze(1)


                            features = torch.cat([features_combined_batch, features_combined_batch_aug], dim=1)
                            out_stage1 = self.model.logits(combined_batch)

                        # Giải phóng hook ngay lập tức
                        # for handle in self.MRFA.remove_handles:
                        #     handle.remove()
                        # self.MRFA.remove_handles.clear()

                        self.model = self.model.train()

                        loss_stage1 = self.criterion(features, combined_labels)
                        loss_stage2_from1 = loss_func(out_stage1, combined_labels)
                        loss = loss_stage1 + self.ELLA_alpha * loss_stage2_from1

                        scaler = torch.cuda.amp.GradScaler()
                        scaler.scale(loss).backward()
                        scaler.step(self.opt)
                        scaler.update()
                        self.opt.zero_grad()

                        # Giải phóng bộ nhớ luôn
                        del features, features_combined_batch, features_combined_batch_aug
                        torch.cuda.empty_cache()

                        # Stage 2
                        for param in self.model.encoder.parameters():
                            param.requires_grad = False
                        for param in self.model.encoder.linear.parameters():
                            param.requires_grad = True
                        stage2_opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd)


                        with torch.cuda.amp.autocast():
                            self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * np.random.rand(mem_x.size(0))).tolist()
                            #self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * (0.7 + 0.3 * np.random.rand(mem_x.size(0)))).tolist()
                            # random lại layer cho các mẫu
                            self.MRFA.perturbation_layers = []
                            self.MRFA.perturbation_layers.extend(np.random.randint(0, len(self.perturb_p), mem_x.size(0)).tolist())
                            out = self.model.logits(combined_batch)
                            loss_stage2 = loss_func(out, combined_labels)

                        scaler.scale(loss_stage2).backward()
                        scaler.step(stage2_opt)
                        scaler.update()
                        stage2_opt.zero_grad()

                        for handle in self.MRFA.remove_handles:
                            handle.remove()
                        self.MRFA.remove_handles.clear()

                    self.buffer.update(batch_x, batch_y)

        self.after_train()
        torch.cuda.empty_cache()