import os
import platform

# File Paths
CURRENT_OS = platform.system()
PROJECT_DIR = '/home/ftian/storage/projects/MFM_exploration/' if CURRENT_OS == 'Linux' else '/Users/tf/Computer_Science/Archive/MFM_exploration/'
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
ANALYSIS_DIR = os.path.join(SRC_DIR, 'analysis')
UTILS_DIR = os.path.join(SRC_DIR, 'utils')
MATLAB_SCRIPT_DIR = os.path.join(UTILS_DIR, 'matlab')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# NEUROMAPS related
NEUROMAPS_SRC_DIR = os.path.join(ANALYSIS_DIR, 'neuromaps')
NEUROMAPS_DATA_DIR = os.path.join(DATA_DIR, 'neuromaps')
DESIKAN_NEUROMAPS_DIR = os.path.join(NEUROMAPS_DATA_DIR, 'desikan')

# Parcellation related
NUM_ROI = 68
NUM_DESIKAN_ROI = 72
IGNORED_DESIKAN_ROI = [1, 5, 37, 41]
IGNORED_DESIKAN_ROI_ZERO_BASED = [0, 4, 36, 40]

# Dataset related
NUM_GROUPS_PNC_AGE = 29
NUM_GROUPS_PNC_COGNITION = 14
