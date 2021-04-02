import multiprocessing as mp
import os
import yaml
import pickle
import re
from datetime import datetime

import numpy as np
from sklearn.linear_model import Lasso


def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]


def getSubName(text, key):
    text = text.split("_")
    text.remove(key)

    return "_".join(text)


def getVariable(T, t_arr, n):
    t = np.tile(np.array([t_arr]).transpose(), (1, n))
    n = np.tile(np.arange(n), (len(t_arr), 1))
    n0 = np.ceil(n / 2)

    out = np.cos((2 * np.pi * n0 * t) / T - (n % 2) * np.pi / 2)
    out[:, 0] = t[:, 0]

    return out


def getDayofYear(filename):

    date = filename.split("_")[4]
    day_of_year = datetime.strptime("-".join([date[0:4], date[4:6], date[6:8]]), '%Y-%m-%d').timetuple().tm_yday

    return day_of_year


def getHarmonics(index, config="./config.yaml"):
    with open(config, "r") as config:
        params = yaml.safe_load(config)["Lasso"]

    aoi = params["aoi"]
    polarizations = params["polarizations"]
    # Harmonics parameter
    T = params["Harmonic_frequency"]
    harPair = params["Harmonic_pairs"]

    dir_ready = os.path.split(params["dir_ready"])[0].format(aoi)
    dir_meta = params["dir_meta"].format(dir_ready, aoi)
    with open(dir_meta, "rb") as dst:
        dst_meta = pickle.load(dst)
    lasso_coefs = np.memmap("tmp.dat",
                            dtype="float64",
                            mode="r+",
                            shape=(2 + harPair * 2, len(polarizations), dst_meta["height"], dst_meta["width"]))

    # Lasso
    coefs_row = np.zeros((2 + harPair * 2, 2, dst_meta["width"]))

    for i in range(len(polarizations)):
        # source files
        dir_ready = params["dir_ready"].format(aoi, aoi, polarizations[i])
        file_ls = [m for m in os.listdir(dir_ready) if m.endswith(".npy")]

        # Get variables and target
        days = np.array([getDayofYear(m) for m in file_ls])
        x = getVariable(T, days, 1 + harPair * 2)
        row = np.array([np.load(os.path.join(dir_ready, m), "r")[index,:] for m in file_ls])
        del file_ls, days

        for j in range(row.shape[1]):
            y = row[:,j]
            try:
                lasso = Lasso(max_iter=10000, alpha=0.2).fit(x, y)
                coef = lasso.coef_
                intercept = lasso.intercept_
            except:
                try:
                    x_new = x[~np.isnan(y)]
                    y_new = y[~np.isnan(y)]
                    lasso = Lasso(max_iter=10000, alpha=0.2).fit(x_new, y_new)
                    coef = lasso.coef_
                    intercept = lasso.intercept_
                except:
                    coef = np.empty((1 + harPair * 2,))
                    coef[:] = np.nan
                    intercept = np.nan
            coefs = np.insert(coef, 0, intercept)
            coefs_row[:, i, j]= coefs
        lasso_coefs[:, :, index, :] = coefs_row

    if (index % 100) == 0:
        print("{} finished".format(index))

