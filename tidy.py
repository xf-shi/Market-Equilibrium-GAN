import os
from os.path import isfile, join, exists
import shutil

def get_files(path):
    f_lst = [f for f in os.listdir(path) if isfile(join(path, f))]
    return f_lst

def get_ts(fname):
    return fname.split("_")[-1].split(".")[0]

def clean(dest_dir):
    model_files = get_files("Models/")
    plot_files = get_files("Plots/")
    model_ts = set([get_ts(x) for x in model_files])
    plot_ts = set([get_ts(x) for x in plot_files])
    overlap_ts = list(model_ts.intersection(plot_ts))
    
    if exists(join(dest_dir, "Models")):
        shutil.rmtree(join(dest_dir, "Models"))
        os.mkdir(join(dest_dir, "Models"))
    if exists(join(dest_dir, "Plots")):
        shutil.rmtree(join(dest_dir, "Plots"))
        os.mkdir(join(dest_dir, "Plots"))
    
    for model_f in model_files:
        if get_ts(model_f) in overlap_ts:
            shutil.copy(join("Models/", model_f), join(dest_dir, "Models"))
    for plot_f in plot_files:
        if get_ts(plot_f) in overlap_ts:
            shutil.copy(join("Plots/", plot_f), join(dest_dir, "Plots"))
    if exists("Logs.tsv"):
        shutil.copy("Logs.tsv", dest_dir)

clean("../Market-Equilibrium-GAN")