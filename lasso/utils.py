import os
import multiprocessing as mp


def multicore(func, args, num_cores = os.cpu_count()):
    """
    This function triggers a parallel processing of specified functions on each element of the args.

    Params:
        func -- (function)
        args -- (list) arguments for specified functions
        num_cores -- (int) number of processes running simultaneously at one time, default is number of virtual cores of
                    device

    """
    pool = mp.Pool(processes=min(len(args), num_cores))
    pool.map(func, args)
    pool.close()
    pool.join()


def square(i):
    print(i*i)