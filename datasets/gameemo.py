import os
from typing import Any, Callable, Dict, Tuple, Union
from torcheeg.datasets.module.base_dataset import BaseDataset
import pandas as pd
from .constant import *

class GAMEEMODataset(BaseDataset):
    def __init__(self, root_path: str = '/mnt/data/original/GAMEEMO',
                 chunk_size: int = 128*2, overlap: int = 128*1, num_channel: int = 14,
                 online_transform: Union[None, Callable] = None,offline_transform: Union[None, Callable] = None,label_transform: Union[None, Callable] = None,
                 io_path: str = os.path.join(os.getcwd(),'PREPROCESSED'), io_size: int = 10485760, io_mode: str = 'lmdb',
                 num_worker: int = 0, verbose: bool = True):
        # pass all arguments to super class
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel,
                  'online_transform': online_transform, 'offline_transform': offline_transform, 'label_transform': label_transform,
                  'io_path': io_path,'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker,'verbose': verbose}
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)
    @staticmethod
    def process_record(file: Any = None, root_path: str = '/mnt/data/GAMEEMO',
                       chunk_size: int = 128*2, overlap: int = 128*1, offline_transform: Union[None, Callable] = None, **kwargs):
        subject_id = int(file[2:4])
        file_path = os.path.join(root_path, file,'Preprocessed EEG Data','.csv format')
        file_list = os.listdir(file_path)
        SAM_path = os.path.join(root_path, file, 'SAM Ratings')
        write_pointer = 0
        # segmentation
        for file_name in file_list:
            trial_id = int(file_name[4])
            csv = pd.read_csv(os.path.join(file_path, file_name), usecols=GAMEEMO_CHANNEL_LIST)
            samples = csv.to_numpy()   # (time, channel)
            samples = samples.swapaxes(0, 1)  # (channel, time)
            # ex) G1.txt
            rating = open(os.path.join(SAM_path, f'G{trial_id}.txt'), 'r')
            rating = rating.readline().strip()
            # record the common meta info
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id, 'emotion':trial_id-1, 
                               'valence': int(rating[1]), 'arousal': int(rating[-1])}
            start_at = 128 * 60
            if chunk_size <= 0: dynamic_chunk_size = samples.shape[1] - start_at
            else:   dynamic_chunk_size = chunk_size
            # chunk with chunk size
            end_at = start_at + dynamic_chunk_size
            # calculate moving step
            step = dynamic_chunk_size - overlap
            while end_at <= samples.shape[1] - 128* 60*3:
                clip_sample = samples[:, start_at:end_at]
                t_eeg = clip_sample
                if not offline_transform is None: t_eeg = offline_transform(eeg=clip_sample)['eeg']
                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1
                # record meta info for each signal
                record_info = {'start_at': start_at, 'end_at': end_at, 'clip_id': clip_id }
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}
                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size
    def set_records(self, root_path: str = '/mnt/data/GAMEEMO', **kwargs):
        return sorted(os.listdir(root_path), key=lambda x : int(x[2:4]))
    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)
        signal = eeg
        label = info
        if self.online_transform: signal = self.online_transform(eeg=eeg)['eeg']
        if self.label_transform:  label = self.label_transform(y=info)['y']
        return signal, label
    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{'root_path': self.root_path,
                'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel,
                'online_transform': self.online_transform, 'offline_transform': self.offline_transform,'label_transform': self.label_transform,
                'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})    
