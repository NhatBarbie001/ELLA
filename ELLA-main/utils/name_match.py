from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.imagenet_subset import ImageNet_Subset
from continuum.dataset_scripts.vfn import VFN
from agents.exp_replay import ExperienceReplay
from agents.scr import SupContrastReplay
from agents.DELTA import DELTA
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update
from utils.buffer.sc_retrieve import Match_retrieve
from utils.buffer.mem_match import MemMatch_retrieve

from agents.ELLA import ELLA



data_objects = {
    'cifar100': CIFAR100,
    'imagenet_subset': ImageNet_Subset,
    'vfn': VFN
}

agents = {
    'ER': ExperienceReplay,
    'SCR': SupContrastReplay,
    'DELTA': DELTA,
    'ELLA': ELLA
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'match': Match_retrieve,
    'mem_match': MemMatch_retrieve

}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'ASER': ASER_update
}

