
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial

class MRFA:
    def __init__(self, with_input_norm=True) -> None:
        self._init_inbatch_properties()
        self.with_input_norm = with_input_norm
        self.perturbations = []
        self.remove_handles = []

    def _init_inbatch_properties(self):
        self.perturbation_layers = []
        self.perturbation_factor = []
        self.perturbation_idices = []
        self.perturbation_idices_inbatch = []

    def feature_augmentation(self, model, convnet, samples, samples_aug, targets, net_type,
                              criterion=None, balanced_loss_stage1=None, for_mem_x_aug=False, ELLAalpha=1):
        # DEBUG: In ra net_type được truyền vào feature_augmentation
        #print(f"DEBUG(MRFA): feature_augmentation called with net_type = {net_type}")
        if net_type == 'resnet32':
            self.get_feature_augmentation(
                model=model,
                convnet=model.encoder,
                samples=samples,
                samples_aug=samples_aug,
                targets=targets,
                num_layers=4, # <-- num_layers is 4 (including linear)
                register_func=register_forward_prehook_resnet32,
                criterion=criterion,
                balanced_loss_stage1=balanced_loss_stage1,
                for_mem_x_aug=for_mem_x_aug,
                ELLAalpha=ELLAalpha
            )
        # elif net_type == 'der_resnet32':
        #     self.get_feature_augmentation(
        #         model=model,
        #         convnet=model.convnets[-1],
        #         samples=samples,
        #         samples_aug=samples_aug,
        #         targets=targets,
        #         num_layers=4,
        #         register_func=register_forward_prehook_resnet32,
        #         criterion=criterion,
        #         balanced_loss_stage1=balanced_loss_stage1,
        #         for_mem_x_aug=for_mem_x_aug
        #     )
        elif net_type == 'resnet18':
            self.get_feature_augmentation(
                model=model,
                convnet=model.encoder,
                samples=samples,
                samples_aug=samples_aug,
                targets=targets,
                num_layers=5, # <-- num_layers is 5 (including linear)
                register_func=register_forward_prehook_resnet18,
                criterion=criterion,
                balanced_loss_stage1=balanced_loss_stage1,
                for_mem_x_aug=for_mem_x_aug,
                ELLAalpha=ELLAalpha
            )
        else:
            raise ValueError(f'Unknown net_type {net_type}.')

    def register_perturb_forward_prehook(self, model, net_type):
        # DEBUG: In ra net_type được truyền vào register_perturb_forward_prehook
        #print(f"DEBUG(MRFA): register_perturb_forward_prehook called with net_type = {net_type}")
        if net_type == 'resnet32':
            self.register_perturb_forward_prehook_layers(model, model.encoder, 4, register_forward_prehook_resnet32)
        elif net_type == 'resnet18':
            self.register_perturb_forward_prehook_layers(model, model.encoder, 5, register_forward_prehook_resnet18) # <-- num_layers được thiết lập là 5 ở đây
        else:
            raise ValueError(f'Unknown net_type {net_type}.')

    def get_feature_augmentation(self, model, convnet, samples, samples_aug, targets, num_layers, register_func,
                                  criterion=None, balanced_loss_stage1=None, for_mem_x_aug=False, ELLAalpha=1):
        # DEBUG: In ra giá trị của num_layers ngay khi hàm bắt đầu
        #print(f"DEBUG(get_feature_augmentation): num_layers = {num_layers}")

        layer_inputs = []
        def get_input_prehook(module, inp):
            # DEBUG: In ra ID của layer và shape của input tensor
            current_layer_id = len(layer_inputs)
            #print(f"DEBUG(get_input_prehook): Layer ID: {current_layer_id}, Input shape: {inp[0].shape}")

            inp[0].requires_grad_(True)  # Đảm bảo input của layer có requires_grad
            inp[0].retain_grad()  # Lưu gradient của input
            layer_inputs.append(inp[0])

        # Đăng ký hook để lưu đầu vào của các layer
        # DEBUG: In ra số lượng hooks được tạo ra
        hooks_to_register = [get_input_prehook] * num_layers
        #print(f"DEBUG(get_feature_augmentation): Registering {len(hooks_to_register)} hooks via register_func.")
        remove_handles = register_func(model, convnet, hooks_to_register)

        # Đặt mô hình ở chế độ train để đảm bảo gradient được theo dõi
        model = model.train()
        # Đảm bảo các tham số mô hình có requires_grad=True
        for param in model.parameters():
            param.requires_grad = True

        # Đặt requires_grad cho tensor đầu vào
        samples.requires_grad_(True)
        samples_aug.requires_grad_(True)

        # Bước 1: Tính features
        features_orig = model.forward(samples).unsqueeze(1)

        #print(f"===========================================----------")
        features_aug = model.forward(samples_aug).unsqueeze(1)
        features = torch.cat([features_orig, features_aug], dim=1)
        #print(f"===========================================----------")
        # Bước 2: Tính loss
        contrastive_loss = criterion(features, targets)
        loss = contrastive_loss + ELLAalpha * balanced_loss_stage1

        if not loss.requires_grad or loss.grad_fn is None:
            raise ValueError("Loss does not have grad_fn or requires_grad=False")

        # Bước 3: Tính và lưu gradient
        model.zero_grad()
        loss.backward()
        #print(f"===========================================----------")

        # Lưu gradient của các layer
        inp_grads = []
        for i, inp in enumerate(layer_inputs):
            if inp.grad is not None:
                inp_grads.append(inp.grad.detach().clone())
                # DEBUG: In ra shape của gradient sau khi backward
                #print(f"DEBUG(get_feature_augmentation): Grad shape for Layer ID {i}: {inp.grad.shape}")
            else:
                print(f"DEBUG(get_feature_augmentation): WARNING: Grad is None for Layer ID {i}. This layer might not be part of the backward path.")

        self.perturbations = inp_grads

        # Kiểm tra NaN trong gradient
        for i, p in enumerate(self.perturbations):
            if torch.isnan(p).any():
                print(f"WARNING: NaN values detected in perturbation {i}!")
                raise ValueError()
            #print(f"Perturbation {i} shape: {p.shape}") # <-- Dòng in shape perturbation gốc

        # Giải phóng hook
        for handle in remove_handles:
            handle.remove()
        self.remove_handles.extend(remove_handles) # Ghi chú: self.remove_handles sẽ được xóa ở agent DELTA

        # Tắt requires_grad để tiết kiệm bộ nhớ
        samples.requires_grad_(False)
        samples_aug.requires_grad_(False)

        # Giải phóng bộ nhớ
        del features, features_orig, features_aug, layer_inputs
        torch.cuda.empty_cache()

    def register_perturb_forward_prehook_layers(self, model, convnet, num_layers, register_func):
        # DEBUG: In ra num_layers và convnet_type khi đăng ký hook perturbation
        #print(f"DEBUG(register_perturb_forward_prehook_layers): num_layers = {num_layers}")

        def perturb_input_prehook_full(module: torch.nn.Module, inp, layer_id):
            # --- CHỈ SỬA ĐOẠN NÀY ĐỂ BỎ QUA LAYER 4 (LINEAR LAYER) ---
            if layer_id == 4: # Layer ID 4 là layer linear cuối cùng
                #print(f"DEBUG(perturb_input_prehook_full): Skipping perturbation for Layer ID {layer_id} (Linear Layer).")
                return (inp[0],) # Trả về input gốc mà không sửa đổi
            # --- KẾT THÚC PHẦN SỬA ĐỔI ---

            if layer_id in self.perturbation_layers:
                inp0 = inp[0].clone()
                p_layers = np.array(self.perturbation_layers)
                p_factor = np.array(self.perturbation_factor)
                p_idices = np.array(self.perturbation_idices)
                p_idices_inbatch = np.array(self.perturbation_idices_inbatch)

                p_index = np.nonzero(p_layers == layer_id)[0]
                if len(self.perturbations) <= layer_id:
                    print(f"ERROR: Layer ID {layer_id} is out of range for perturbations list of size {len(self.perturbations)}")
                    return (inp0,)

               
                # === THÊM PRINT STATEMENTS VÀO ĐÂY ===
                # Áp dụng perturbation
                num_new_axises = len(self.perturbations[layer_id].size()) - 1
                if self.with_input_norm:

                    # Thành phần 1: Chuẩn hóa đầu vào
                    input_norm_val = (inp0.data[p_idices_inbatch[p_index]].view(len(p_index), -1).norm(dim=-1) ** 2)
                    input_norm_val_expanded = input_norm_val[:, *(None,) * num_new_axises]
                   

                    # Thành phần 2: Gradient (perturbation_slice)
                    perturbation_slice = self.perturbations[layer_id][p_idices[p_index]]
                  

                    # Thành phần 3: Hệ số nhiễu (p_factor_expanded)
                    p_factor_batch_slice = p_factor[p_index]
                    p_factor_expanded = torch.from_numpy(p_factor_batch_slice).float()[:, *(None,) * num_new_axises].to(inp0.device)
                   

                    # Tính toán tổng nhiễu
                    perturb = input_norm_val_expanded * perturbation_slice * p_factor_expanded
                    # print(f"  - Final Perturb (value) first 5: {perturb.flatten()[:5].tolist()}")
                    # print(f"  - Final Perturb (shape): {perturb.shape}")
                    #perturb = (inp0.data[p_idices_inbatch[p_index]].view(len(p_index), -1).norm(dim=-1) ** 2)[:, *(None,) * num_new_axises] * self.perturbations[layer_id][p_idices[p_index]] * torch.from_numpy(p_factor[p_index]).float()[:, *(None,) * num_new_axises].to(inp0.device)
                else:
                    print(f"dayyyyyy roiiiiiiiiiiiii2222222222222222222")
                    perturb = self.perturbations[layer_id][p_idices[p_index]] * torch.from_numpy(p_factor[p_index]).float()[:, *(None,) * num_new_axises].to(inp0.device)


                inp0[p_idices_inbatch[p_index]] += perturb

                # Lấy phần input sau khi đã perturb
                input_slice_after = inp0[p_idices_inbatch[p_index]]
                # print(f"Input slice (after perturbation, first 5 elements): {input_slice_after.flatten()[:5].tolist()}")
                # === KẾT THÚC THÊM PRINT STATEMENTS ===

                return (inp0,)
            return (inp[0],) # Quan trọng: nếu không có perturbation, vẫn trả về input gốc

        hooks = [partial(perturb_input_prehook_full, layer_id=i) for i in range(num_layers)]
        # DEBUG: In ra số lượng hooks được tạo để đăng ký
        #print(f"DEBUG(register_perturb_forward_prehook_layers): Number of hooks created to register: {len(hooks)}")
        self.remove_handles.extend(register_func(model, convnet, hooks))

def register_forward_prehook_resnet32(model, convnet, hooks):
    #print(f"DEBUG(register_forward_prehook_resnet32): called with len(hooks) = {len(hooks)}")
    remove_handles = []

    remove_handle_stage_1 = convnet.stage_1.register_forward_pre_hook(hooks[0])
    remove_handles.append(remove_handle_stage_1)
    #print(f"DEBUG(register_forward_prehook_resnet32): Hooked convnet.stage_1 (index 0), current handles: {len(remove_handles)}")

    remove_handle_stage_2 = convnet.stage_2.register_forward_pre_hook(hooks[1])
    remove_handles.append(remove_handle_stage_2)
    #print(f"DEBUG(register_forward_prehook_resnet32): Hooked convnet.stage_2 (index 1), current handles: {len(remove_handles)}")

    remove_handle_stage_3 = convnet.stage_3.register_forward_pre_hook(hooks[2])
    remove_handles.append(remove_handle_stage_3)
    #print(f"DEBUG(register_forward_prehook_resnet32): Hooked convnet.stage_3 (index 2), current handles: {len(remove_handles)}")

    # remove_handle_fc = convnet.linear.register_forward_pre_hook(hooks[3])
    # remove_handles.append(remove_handle_fc)
    #print(f"DEBUG(register_forward_prehook_resnet32): Hooked model.fc (index 3), current handles: {len(remove_handles)}")

    #print(f"DEBUG(register_forward_prehook_resnet32): Returning {len(remove_handles)} handles.")
    return remove_handles

def register_forward_prehook_resnet18(model, convnet, hooks):
    #print(f"DEBUG(register_forward_prehook_resnet18): called with len(hooks) = {len(hooks)}")
    remove_handles = []

    # DEBUG: Thêm print cho từng layer được hook
    remove_handle_layer_1 = convnet.layer1.register_forward_pre_hook(hooks[0])
    remove_handles.append(remove_handle_layer_1)
    #print(f"DEBUG(register_forward_prehook_resnet18): Hooked convnet.layer1 (index 0), current handles: {len(remove_handles)}")

    remove_handle_layer_2 = convnet.layer2.register_forward_pre_hook(hooks[1])
    remove_handles.append(remove_handle_layer_2)
    #print(f"DEBUG(register_forward_prehook_resnet18): Hooked convnet.layer2 (index 1), current handles: {len(remove_handles)}")

    remove_handle_layer_3 = convnet.layer3.register_forward_pre_hook(hooks[2])
    remove_handles.append(remove_handle_layer_3)
    #print(f"DEBUG(register_forward_prehook_resnet18): Hooked convnet.layer3 (index 2), current handles: {len(remove_handles)}")

    remove_handle_layer_4 = convnet.layer4.register_forward_pre_hook(hooks[3])
    remove_handles.append(remove_handle_layer_4)
    #print(f"DEBUG(register_forward_prehook_resnet18): Hooked convnet.layer4 (index 3), current handles: {len(remove_handles)}")

    # remove_handle_linear = convnet.linear.register_forward_pre_hook(hooks[4])
    # remove_handles.append(remove_handle_linear)
    #print(f"DEBUG(register_forward_prehook_resnet18): Hooked convnet.linear (index 4), current handles: {len(remove_handles)}")

    #print(f"DEBUG(register_forward_prehook_resnet18): Returning {len(remove_handles)} handles.")
    return remove_handles