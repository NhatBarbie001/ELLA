import numpy as np
from scipy.stats import sem
import scipy.stats as stats
import json

def compute_performance(end_task_acc_arr, params, time):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))
    write_accuracy = (np.sum(np.tril(end_task_acc_arr), axis=2) /(np.arange(n_tasks) + 1)).tolist()
    
    if params.write_file:
        if params.file_name is not None:
            filename = params.file_name
        else:
            filename = 'out_files.txt'

        settings_str = str(params.agent)+" "+str(params.update)+" "+str(params.retrieve)+" "+str(params.data)+" mem_size: "+str(params.mem_size)+" nc_first_task: "+str(params.nc_first_task)+" num_tasks: "+str(params.num_tasks)+" exemplars: "+str(params.eps_mem_batch)+" temp: "+str(params.temp)+" head: "+str(params.head)+" seed:"+str(params.seed)
        with open(filename, 'a') as outfile:
            json.dump(settings_str, outfile)
            outfile.write('\n')
            json.dump(write_accuracy, outfile)
            outfile.write('\n')
            json.dump(str(avg_fgt), outfile) 
            outfile.write('\n')
            json.dump("time: "+str(time), outfile)
            outfile.write('\n')


    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_fgt, avg_acc, avg_bwtp, avg_fwt




def single_run_avg_end_fgt(acc_array):
    best_acc = np.max(acc_array, axis=1)
    end_acc = acc_array[-1]
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets)
    return avg_fgt
