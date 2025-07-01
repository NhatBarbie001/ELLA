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

# Thêm import MRFA
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

class DELTA(ContinualLearner):
    def __init__(self, model, opt, params):
        super(DELTA, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters

        # Khởi tạo các thuộc tính MRFA
        # perturb_p: mảng các hệ số perturbation cho các layer
        self.perturb_p = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
        # disable_perturb: bật/tắt perturbation
        #self.disable_perturb = params.get('disable_perturb', False)

        # num_augmem: số lần lặp lại augmented memory
        #self.num_augmem = params.get('num_augmem', 1)

        # perturb_all: áp dụng perturbation cho tất cả samples hay chỉ memory samples
        #self.perturb_all = params.get('perturb_all', False)

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
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        #self.model = self.model.train()
        # setup tracker
        losses = AverageMeter()
        losses_stage2 = AverageMeter()
        acc_batch = AverageMeter()


        #print(f"\nnew Task:==================================================================\n")
        cur_batch = 0
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):

                cur_batch += 1
                #print(f"\nnew Batch: {cur_batch}  =======================\n")
                distribution_vector = np.zeros(self.class_size, dtype='float')
                loss_func = None
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                # print(f"\n=== Training Batch Info ===")
                # print(f"Current batch size: {batch_x.size(0)}")
                # print(f"Input shape: {batch_x.shape}")
                # print(f"Unique labels in batch: {torch.unique(batch_y).tolist()}")
                # print(f"Label distribution: {torch.bincount(batch_y).tolist()}")

                #Stage 1
                for j in range(self.mem_iters):

                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    # print(f"\n=== Memory Retrieval Info (Iter {j+1}/{self.mem_iters}) ===")
                    # print(f"Retrieved memory size: {mem_x.size(0)}")
                    for param in self.model.encoder.parameters():
                        param.requires_grad = True
                    if mem_x.size(0) > 0:
                        #print(f"Memory input shape: {mem_x.shape}")
                        #print(f"Unique labels in memory: {torch.unique(mem_y).tolist()}")
                        #print(f"Memory label distribution: {torch.bincount(mem_y).tolist()}")
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)

                        distribution_vector = update_distribution(distribution_vector, torch.cat((batch_y, mem_y)))
                        loss_func = BalancedSoftmaxLoss(np.array(distribution_vector), self.tasks, self.class_size, which_task).cuda()

                        combined_batch = torch.cat((batch_x, mem_x))
                        combined_labels = torch.cat((batch_y, mem_y))
                        # print(f"\n=== Combined Batch Info ===")
                        # print(f"Combined batch size: {combined_batch.size(0)}")
                        # print(f"Combined batch shape: {combined_batch.shape}")
                        # print(f"Combined labels shape: {combined_labels.shape}")
                        #print(f"Unique labels in combined batch: {torch.unique(combined_labels).tolist()}")
                        #print(f"Combined label distribution: {torch.bincount(combined_labels).tolist()}")
                        
                        combined_batch_aug = self.transform(combined_batch)
                        # print(f"\n=== Augmented Batch Info ===")
                        # print(f"Augmented batch size: {combined_batch_aug.size(0)}")
                        # print(f"Augmented batch shape: {combined_batch_aug.shape}")
                        #print(f"Augmented batch value range: [{combined_batch_aug.min().item():.4f}, {combined_batch_aug.max().item():.4f}]")
                        #print(f"Augmented batch mean: {combined_batch_aug.mean().item():.4f}")
                        #print(f"Augmented batch std: {combined_batch_aug.std().item():.4f}")

                        #-----------------------------------------------------------------------------------------
                        #TRIỂN KHAI áp dụng nhiễu tại đây
                        # tốt rồi, dl trong 1 batch không bị shuffle
                        # 
                        self.MRFA._init_inbatch_properties()
                        mem_x_aug = combined_batch_aug[batch_x.size(0):batch_x.size(0) + mem_x.size(0)]
                        # # print(f"\n=== Debug Compute MRFA Perturbation ===")
                        # # print(f"Memory batch size: {mem_x.size(0)}")
                        # # print(f"Augmented memory batch size: {mem_x_aug.size(0)}")

                        perturb_mask = torch.zeros(batch_x.size(0) + mem_x.size(0), dtype=torch.bool)
                        perturb_mask[batch_x.size(0):] = True  # Đánh dấu các sample từ memory
                        # #print(f"Number of samples to perturb: {perturb_mask.sum().item()}")

                        # # Lưu thông tin perturbation
                        self.MRFA.perturbation_idices.extend(np.arange(mem_x.size(0)).tolist())
                        self.MRFA.perturbation_idices_inbatch.extend(perturb_mask.nonzero().flatten().tolist())
                        self.MRFA.perturbation_layers.extend(np.random.randint(0, len(self.perturb_p), mem_x.size(0)).tolist())
                        self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * np.random.rand(mem_x.size(0))).tolist()
                        
                        # # print("\nPerturbation details:")
                        # # print(f"Number of perturbation indices: {len(self.MRFA.perturbation_idices)}")
                        # # print(f"Number of perturbation layers: {len(self.MRFA.perturbation_layers)}")
                        # # print(f"Perturbation factors range: [{min(self.MRFA.perturbation_factor):.6f}, {max(self.MRFA.perturbation_factor):.6f}]")

                        # ###########################################################################################################################
                        # # gọi hàm feature_augmentation để lưu grad ở các layer của mem_x vào MRFA.perturbations
                        # #print("\nComputing perturbations for original memory samples...")
                        # self.MRFA.feature_augmentation(
                        #     model=self.model,
                        #     samples=mem_x,
                        #     samples_aug=mem_x_aug,
                        #     targets=mem_y,
                        #     net_type=self.convnet_type,
                        #     criterion=self.criterion,
                        #     for_mem_x_aug=False
                        # )
                        
                        # đăng ký hook(train) vào model
                        #self.MRFA.register_perturb_forward_prehook(self.model, self.convnet_type)
                        features_combined_batch = self.model.forward(combined_batch).unsqueeze(1)
                        # if len(self.MRFA.remove_handles) > 0:
                        #     for handle in self.MRFA.remove_handles:
                        #         handle.remove()
                        #     self.MRFA.remove_handles.clear()
                        

                        # #print("\nComputing perturbations for augmented memory samples...")
                        # self.MRFA.feature_augmentation(
                        #     model=self.model,
                        #     samples=mem_x,
                        #     samples_aug=mem_x_aug,
                        #     targets=mem_y,
                        #     net_type=self.convnet_type,
                        #     criterion=self.criterion,
                        #     for_mem_x_aug=True
                        # )
                        # #self.model = self.model.train()
                        # # đăng ký hook(train) vào model
                        # self.MRFA.register_perturb_forward_prehook(self.model, self.convnet_type)
                        features_combined_batch_aug = self.model.forward(combined_batch_aug).unsqueeze(1)
                        # if len(self.MRFA.remove_handles) > 0:
                        #     for handle in self.MRFA.remove_handles:
                        #         handle.remove()
                        #     self.MRFA.remove_handles.clear()
                        
                        ###########################################################################################################################
                        
                        #self.model = self.model.train()
                        #print("\nApplying perturbations to combined batch...")
                        features = torch.cat([features_combined_batch, features_combined_batch_aug], dim=1)
                        #print(f"Final features shape: {features.shape}")
                        self.model = self.model.train()
                        out_stage1 = self.model.logits(torch.cat([combined_batch, combined_batch_aug]))
                        #print(f"Stage 1 output shape: {out_stage1.shape}")

                        # loss này dùng cho học biểu diễn 
                        loss_stage1 = self.criterion(features, combined_labels)
                        # loss này dùng cho học cân bằng
                        loss_stage2_from1 = loss_func(out_stage1, torch.cat([combined_labels, combined_labels]))
                        # tổng loss
                        loss = loss_stage1 + loss_stage2_from1
                        # print(f"\nLoss values:")
                        # print(f"Stage 1 loss: {loss_stage1.item():.4f}")
                        # print(f"Stage 2 loss: {loss_stage2_from1.item():.4f}")
                        # print(f"Total loss: {loss.item():.4f}")
                        # print(f"\nend Batch  =======================\n")


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
        if len(self.MRFA.remove_handles) > 0:
            for handle in self.MRFA.remove_handles:
                handle.remove()
            self.MRFA.remove_handles.clear()
