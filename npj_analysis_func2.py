home_dir = "/home/dhkim/"
data_dir = "/home/dhkim/data/"
ML_dir = "/home/dhkim/research/Machine_Learning/MLresults/"


import re
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
import yaml
from matplotlib.ticker import FormatStrFormatter

from dhkpython.PLOT import pcolor_dhk
