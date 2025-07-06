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

class DELTA(ContinualLearner):
    def __init__(self, model, opt, params):
        super(DELTA, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
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
        print(f"[DELTA] Task {which_task}: Training with {len(x_train)} samples.")
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        losses_stage2 = AverageMeter()
        acc_batch = AverageMeter()
        for ep in range(self.epoch):
            
            for i, batch_data in enumerate(train_loader):
                distribution_vector = np.zeros(self.class_size, dtype='float')
                loss_func = None
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #Stage 1
                for j in range(self.mem_iters):
                    print(f"[DELTA][Batch {i}] batch_x: {batch_x.size(0)}, mem_x: {mem_x.size(0)}, total: {batch_x.size(0) + mem_x.size(0)}")
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    # Unfreeze all layers
                    for param in self.model.encoder.parameters():
                        param.requires_grad = True
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)

                        distribution_vector = update_distribution(distribution_vector, torch.cat((batch_y, mem_y)))
                        loss_func = BalancedSoftmaxLoss(np.array(distribution_vector), self.tasks, self.class_size, which_task).cuda()

                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_batch_aug = self.transform(combined_batch)
                        # Vì loss để tối ưu model sẽ là loss sau khi đã augment, nên ta sẽ augment ở đây
                        # bây giờ ta có hai hướng:
                        # 1. Ta chỉ sử dụng loss tương phản để augment feature, tăng cường quá trình học biểu diễn (representation learning)
                        # 2. Ta sử dụng loss tổng loss tương phản và cân bằng để augment feature,
                        #    + Ta có thể tính được loss tương phản 
                        #    + Nhưng làm thế nào để tính được loss cân bằng? (cần ...) anyway







                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        out_stage1 = self.model.logits(torch.cat([combined_batch, combined_batch_aug]))

                        # loss này dùng cho học biểu diễn 
                        loss_stage1 = self.criterion(features, combined_labels)
                        # loss này dùng cho học cân bằng
                        loss_stage2_from1 = loss_func(out_stage1, torch.cat([combined_labels, combined_labels]))
                        # tổng loss
                        loss = loss_stage1 + loss_stage2_from1



                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                        #stage 2
                        
                        # Freeze all layers
                        for param in self.model.encoder.parameters():
                            param.requires_grad = False
                        # Unfreeze the last layer by setting requires_grad to True for its parameters
                        for param in self.model.encoder.linear.parameters():
                            param.requires_grad = True

                        stage2_opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        out = self.model.logits(combined_batch)
                        loss_stage2 = loss_func(out, combined_labels)
                        # losses_stage2.update(loss_stage2, batch_y.size(0))
                        stage2_opt.zero_grad()
                        loss_stage2.backward()
                        stage2_opt.step()

                # # Stage 2
                # for j in range(self.mem_iters):
                #     mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    
                #     if mem_x.size(0) > 0:
                       
                #         distribution_vector = update_distribution(distribution_vector, torch.cat((batch_y, mem_y)))
                #         loss_func = BalancedSoftmaxLoss(np.array(distribution_vector), self.tasks, self.class_size, which_task).cuda()
                #         # Freeze all layers
                #         for param in self.model.encoder.parameters():
                #             param.requires_grad = False
                #         # Unfreeze the last layer by setting requires_grad to True for its parameters
                #         for param in self.model.encoder.linear.parameters():
                #             param.requires_grad = True

                #         stage2_opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

                #         mem_x = maybe_cuda(mem_x, self.cuda)
                #         mem_y = maybe_cuda(mem_y, self.cuda)
                #         combined_batch = torch.cat((mem_x, batch_x))
                #         combined_labels = torch.cat((mem_y, batch_y))
                #         out = self.model.logits(combined_batch)
                #         loss_stage2 = loss_func(out, combined_labels)
                #         # losses_stage2.update(loss_stage2, batch_y.size(0))
                #         stage2_opt.zero_grad()
                #         loss_stage2.backward()
                #         stage2_opt.step()
                #         # combined_batch_aug = self.transform(combined_batch)
                #         # features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                #         # loss = self.criterion(features, combined_labels)
                #         # losses.update(loss, batch_y.size(0))
                #         # self.opt.zero_grad()
                #         # loss.backward()
                #         # self.opt.step()



                # update mem
                self.buffer.update(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                                .format(i, losses.avg(), acc_batch.avg())
                        )
        self.after_train()
