# ELLA: Enhancing Long-tailed Online Continual Learning with Layer Augmentation


Official repository of 



## Requirements
![](https://img.shields.io/badge/python-3.7-green.svg)

![](https://img.shields.io/badge/torch-1.5.1-blue.svg)
![](https://img.shields.io/badge/torchvision-0.6.1-blue.svg)
![](https://img.shields.io/badge/PyYAML-5.3.1-blue.svg)
![](https://img.shields.io/badge/scikit--learn-0.23.0-blue.svg)
----

Installing packages
```sh
pip install kornia
```

## Datasets 

### Online Class Incremental
- CIFAR-100 (Download from torchvision)
- VFN dataset (Downloadable from https://lorenz.ecn.purdue.edu/~vfn/): In [annotonations.txt](annotations.txt), ... We deleted the paths to images that belong to the 8 folders with the fewest photos.
- The long-tailed version of the datasets will be created according to fixed distributions.

## Algorithms 

* ASER: Adversarial Shapley Value Experience Replay(**AAAI, 2021**) [[Paper]](https://arxiv.org/abs/2009.00093)
* iCaRL: Incremental Classifier and Representation Learning (**CVPR, 2017**) [[Paper]](https://arxiv.org/abs/1611.07725)
* ER: Experience Replay (**ICML Workshop, 2019**) [[Paper]](https://arxiv.org/abs/1902.10486)
* MIR: Maximally Interfered Retrieval (**NeurIPS, 2019**) [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
* GSS: Gradient-Based Sample Selection (**NeurIPS, 2019**) [[Paper]](https://arxiv.org/pdf/1903.08671.pdf)
* SCR: Supervised Contrastive Replay (**CVPR Workshop, 2021**) [[Paper]](https://arxiv.org/abs/2103.13885) 
* DVC: Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency (**CVPR 2022**)[[GitHub]](https://github.com/YananGu/DVC)
* OnPRO: Online Prototype Learning (**ICCV 2023**)[[GitHub]](https://github.com/weilllllls/OnPro)
* PRS: Imbalanced Continual Learning with Partioning Reservoir Sampling (**ECCV 2020**)[[GitHub]](https://github.com/cdjkim/PRS)
* CBRS: Online Continual Learning from Imbalanced Data (**ICML 2020** )[[Paper]](https://dl.acm.org/doi/10.5555/3524938.3525120)
* DELTA: Decoupling Long-Tailed Online Continual Learning  [DELTA: Decoupling Long-Tailed Online Continual Learning](https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Raghavan_DELTA_Decoupling_Long-Tailed_Online_Continual_Learning_CVPRW_2024_paper.pdf) (CVPRW 2024) Workshop - CLVision 2024)


## Run commands
Detailed descriptions of options can be found in [general_main.py](general_main.py)

### Sample commands to run algorithms on Split-CIFAR100
```shell
#DELTA
python general_main.py --data cifar100 --agent ELLA --ELLA_alpha 0.7 --ELLA_beta 0.1 --mem_size 1000 --num_tasks 20 --cl_type nc --retrieve random  --update random --head None --temp 0.09 --verbose False --lt True --eps_mem_batch 32 --randomize_seed True --cuda_device 4 --write_file False  --file_name 'test.txt'  --randomize_seed True
```

### Other existing methods as part of the repository
```shell
#ER
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000 --num_tasks 20 --lt True --eps_mem_batch 32 --nc_first_task 5 --randomize_seed True --cuda_device 4 --write_file False  --file_name 'test.txt'  --randomize_seed True

#MIR
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 5000 --num_tasks 20 --lt True --eps_mem_batch 32 --nc_first_task 5 --randomize_seed True --cuda_device 4 --write_file False  --file_name 'test.txt'  --randomize_seed True

#GSS
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update GSS  --gss_mem_strength 20 --mem_size 5000--num_tasks 20 --lt True --eps_mem_batch 32 --nc_first_task 5 --randomize_seed True --cuda_device 4 --write_file False  --file_name 'test.txt'  --randomize_seed True

#ASER
python general_main.py --data cifar100 --cl_type nc --agent ER --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3 --num_tasks 20 --lt True --eps_mem_batch 32 --nc_first_task 5 --randomize_seed True --cuda_device 4 --write_file False  --file_name 'test.txt'  --randomize_seed True

#SCR
python general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 5000 --head mlp --temp 0.07 --eps_mem_batch 100 --num_tasks 20 --lt True --eps_mem_batch 32 --nc_first_task 5 --randomize_seed True --cuda_device 4 --write_file False  --file_name 'test.txt'  --randomize_seed True
```

## Repo Structure & Description
    ├──agents                       #Files for different algorithms
        ├──base.py                      #Abstract class for algorithms
        ├──ELLA.py                     #File for our method - ELLA
        ├──exp_replay.py                #File for ER, MIR and GSS
        ├──scr.py                       #File for SCR
    
    ├──continuum                    #Files for create the data stream objects
        ├──dataset_scripts              #Files for processing each specific dataset
            ├──dataset_base.py              #Abstract class for dataset
            ├──cifar100.py                  #File for making CIFAR100-LT
            ├──imagenet_subset.py           #File for Imagenet subset
            ├──vfn.py                       #File for making VFN-LT

        ├──continuum.py             
        ├──data_utils.py
    
    ├──models                       #Files for backbone models
            ├──...
        ├──pretrained.py                #Files for pre-trained models
        ├──resnet.py                    #Files for ResNet
    
    ├──utils                        #Files for utilities
        ├──buffer                       #Files related to buffer
            ├──aser_retrieve.py             #File for ASER retrieval
            ├──aser_update.py               #File for ASER update
            ├──aser_utils.py                #File for utilities for ASER
            ├──buffer.py                    #Abstract class for buffer
            ├──buffer_utils.py              #General utilities for all the buffer files
            ├──gss_greedy_update.py         #File for GSS update
            ├──mir_retrieve.py              #File for MIR retrieval
            ├──random_retrieve.py           #File for random retrieval
            ├──reservoir_update.py          #File for random update

## Citation 

If you use this paper/code in your research, please consider citing us:




## Contact




## Note
Implementation is based on the repository (https://github.com/RaptorMai/online-continual-learning).
