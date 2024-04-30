from torcheeg.datasets.constants.utils import (format_channel_location_dict, format_region_channel_list)
from torcheeg.datasets.constants.region_1020 import GENERAL_REGION_LIST 

DEAP_CHANNEL_LIST = ['FP1', 'AF3', 'F3', 'F7', 
                     'FC5', 'FC1', 'C3', 'T7', 
                     'CP5', 'CP1', 'P3','P7', 
                     'PO3', 'O1', 'OZ', 'PZ',
                     'FP2', 'AF4', 'FZ', 'F4', 
                     'F8', 'FC6', 'FC2', 'CZ',
                     'C4', 'T8', 'CP6', 'CP2', 
                     'P4', 'P8', 'PO4', 'O2']
DEAP_LOCATION_LIST = [['-', '-', '-', 'FP1', '-', 'FP2', '-', '-', '-'],
                      ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
                      ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
                      ['-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-'],
                      ['T7', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T8'],
                      ['-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-'],
                      ['P7', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'P8'],
                      ['-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-'],
                      ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-']]
DEAP_CHANNEL_LOCATION_DICT = format_channel_location_dict(DEAP_CHANNEL_LIST, DEAP_LOCATION_LIST)
DEAP_GENERAL_REGION_LIST = format_region_channel_list(DEAP_CHANNEL_LIST, GENERAL_REGION_LIST)
DEAP_SUBNUM = 32
DEAP_sfeq = 128; DEAP_l_freq, DEAP_h_freq = 4, 45;

GAMEEMO_CHANNEL_LIST = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8'] # 14 channels
GAMEEMO_LOCATION_LIST = [['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                        ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
                        ['F7', '-', 'F3', '-', '-', '-', 'F4', '-', 'F8'],
                        ['-', 'FC5', '-', '-', '-', '-', '-', 'FC6', '-'],
                        ['T7', '-', '-', '-', '-', '-', '-', '-', 'T8'],
                        ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                        ['P7', '-', '-', '-', '-', '-', '-', '-', 'P8'],
                        ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                        ['-', '-', '-', 'O1', '-', 'O2', '-', '-', '-']]
GAMEEMO_CHANNEL_LOCATION_DICT = format_channel_location_dict(GAMEEMO_CHANNEL_LIST, GAMEEMO_LOCATION_LIST)
GAMEEMO_GENERAL_REGION_LIST = format_region_channel_list(GAMEEMO_CHANNEL_LIST, GENERAL_REGION_LIST)
GAMEEMO_SUBNUM = 28