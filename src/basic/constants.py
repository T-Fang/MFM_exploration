import os
import platform

# File Paths
CURRENT_OS = platform.system()
PROJECT_PATH = '/home/ftian/storage/projects/MFM_exploration/' if CURRENT_OS == 'Linux' else '/Users/tf/Computer_Science/Archive/MFM_exploration/'
LOG_PATH = os.path.join(PROJECT_PATH, 'logs/')
UTILS_PATH = os.path.join(PROJECT_PATH, 'src/utils/')
MATLAB_SCRIPT_PATH = os.path.join(UTILS_PATH, 'matlab/')
