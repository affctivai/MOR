import os
import scipy.io as scio
from typing import Any, Callable, Dict, Tuple, Union
from torcheeg.datasets.module.base_dataset import BaseDataset

class DEAPDataset(BaseDataset):
    def __init__(self, root_path: str = '/mnt/data/original/DEAP/data_preprocessed_matlab',
                 chunk_size: int = 128*4, overlap: int = 128*2, num_channel: int = 32,
                 num_baseline: int = 6, baseline_chunk_size: int = 128,
                 online_transform: Union[None, Callable] = None,offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None, after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None, after_subject: Union[Callable, None] = None,
                 io_path: str = os.path.join(os.getcwd(), 'tmp_out'),
                 io_size: int = 10485760, io_mode: str = 'lmdb', num_worker: int = 0,
                 verbose: bool = True, in_memory: bool = False):
        # pass all arguments to super class
        params = {'root_path': root_path,
            'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel,
            'num_baseline': num_baseline, 'baseline_chunk_size': baseline_chunk_size,
            'online_transform': online_transform, 'offline_transform': offline_transform,'label_transform': label_transform,
            'before_trial': before_trial,'after_trial': after_trial,'after_session': after_session,'after_subject': after_subject,
            'io_path': io_path,'io_size': io_size,'io_mode': io_mode,
            'num_worker': num_worker, 'verbose': verbose,'in_memory': in_memory}
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)
    @staticmethod
    def process_record(file: Any = None, root_path: str = '/mnt/data/original/DEAP/data_preprocessed_matlab',
                   chunk_size: int = 128*4, overlap: int = 128*2, num_channel: int = 32, 
                   num_baseline: int = 6, baseline_chunk_size: int = 128,
                   before_trial: Union[None, Callable] = None, after_trial: Union[Callable, None] = None,
                   offline_transform: Union[None, Callable] = None, **kwargs):
        file_name = file  # an element from file name list
        pkl_data = scio.loadmat(os.path.join(root_path, file_name))

        samples = pkl_data['data']  # trial(40), channel(32), timestep(63*128)
        labels = pkl_data['labels']
        subject_id = int(file_name[1:3])

        write_pointer = 0
        for trial_id in range(len(samples)):
            # extract baseline signals
            trial_samples = samples[trial_id, :num_channel]  # channel(32), timestep(63*128)
            if before_trial:  trial_samples = before_trial(trial_samples)

            trial_baseline_sample = trial_samples[:, :baseline_chunk_size * num_baseline]  # channel(32), timestep(3*128)
            trial_baseline_sample = trial_baseline_sample.reshape(num_channel, num_baseline, baseline_chunk_size).mean(axis=1)  # channel(32), timestep(128)

            # record the common meta info
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id + 1}
            trial_rating = labels[trial_id]
            for label_index, label_name in enumerate(['valence', 'arousal', 'dominance', 'liking']):
                trial_meta_info[label_name] = trial_rating[label_index]

            start_at = baseline_chunk_size * num_baseline
            if chunk_size <= 0:  dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:   dynamic_chunk_size = chunk_size

            # chunk with chunk size
            end_at = start_at + dynamic_chunk_size
            # calculate moving step
            step = dynamic_chunk_size - overlap

            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:, start_at:end_at]

                t_eeg = clip_sample
                t_baseline = trial_baseline_sample

                if not offline_transform is None:
                    t = offline_transform(eeg=clip_sample, baseline=trial_baseline_sample)
                    t_eeg = t['eeg'];  t_baseline = t['baseline']
                # put baseline signal into IO
                if not 'baseline_id' in trial_meta_info:
                    trial_base_id = f'{file_name}_{write_pointer}'
                    yield {'eeg': t_baseline, 'key': trial_base_id}
                    write_pointer += 1
                    trial_meta_info['baseline_id'] = trial_base_id
                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1
                # record meta info for each signal
                record_info = {'start_at': start_at, 'end_at': end_at,  'clip_id': clip_id }
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}
                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self, root_path: str = '/mnt/data/DEAP/data_preprocessed_matlab', **kwargs):
        return sorted(os.listdir(root_path), key=lambda x : int(x[1:3]))

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)
        baseline_index = str(info['baseline_id'])
        baseline = self.read_eeg(eeg_record, baseline_index)
        signal = eeg
        label = info
        if self.online_transform: signal = self.online_transform(eeg=eeg, baseline=baseline)['eeg']
        if self.label_transform: label = self.label_transform(y=info)['y']
        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{'root_path': self.root_path,
                'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel,
                'num_baseline': self.num_baseline, 'baseline_chunk_size': self.baseline_chunk_size,
                'online_transform': self.online_transform, 'offline_transform': self.offline_transform, 'label_transform': self.label_transform,
                'before_trial': self.before_trial, 'after_trial': self.after_trial,
                'num_worker': self.num_worker, 'verbose': self.verbose,'io_size': self.io_size})


