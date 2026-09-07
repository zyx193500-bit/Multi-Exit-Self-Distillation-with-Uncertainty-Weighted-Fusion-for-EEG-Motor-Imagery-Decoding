from pathlib import Path
from typing import Optional

import mne
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4


ROOT = Path(__file__).resolve().parents[1]
BCIC2B_OFFICIAL_ROOT = ROOT / "data_cache" / "bcic2b_official"
BCIC2B_OFFICIAL_GDF_DIR = BCIC2B_OFFICIAL_ROOT / "gdf"
BCIC2B_OFFICIAL_LABEL_DIR = BCIC2B_OFFICIAL_ROOT / "true_labels"


class BCICIV2b(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2
    eeg_channels = ["EEG:C3", "EEG:Cz", "EEG:C4"]
    train_sessions = ["01T", "02T", "03T"]
    test_sessions = ["04E", "05E"]
    task_duration_seconds = 4.5
    cue_offset_seconds = 3.0

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    @classmethod
    def _session_gdf_path(cls, subject_id: int, session: str) -> Path:
        return BCIC2B_OFFICIAL_GDF_DIR / f"B{subject_id:02d}{session}.gdf"

    @classmethod
    def _session_label_path(cls, subject_id: int, session: str) -> Path:
        return BCIC2B_OFFICIAL_LABEL_DIR / f"B{subject_id:02d}{session}.mat"

    @classmethod
    def _has_local_official_subject_data(cls, subject_id: int) -> bool:
        for session in cls.train_sessions + cls.test_sessions:
            if not cls._session_gdf_path(subject_id, session).exists():
                return False
            if not cls._session_label_path(subject_id, session).exists():
                return False
        return True

    @classmethod
    def _window_spec(cls, sfreq: float, preprocessing_dict: dict):
        start = float(preprocessing_dict["start"])
        stop = float(preprocessing_dict["stop"])
        window_duration_seconds = cls.task_duration_seconds + stop - start
        if window_duration_seconds <= 0:
            raise ValueError(
                f"Invalid BCIC2b window configuration: start={start}, stop={stop}, "
                f"duration={window_duration_seconds}"
            )
        start_offset_samples = int(round((cls.cue_offset_seconds + start) * sfreq))
        window_size = int(round(window_duration_seconds * sfreq))
        return start_offset_samples, window_size

    @classmethod
    def _load_session_from_official_files(cls, subject_id: int, session: str, preprocessing_dict: dict):
        gdf_path = cls._session_gdf_path(subject_id, session)
        label_path = cls._session_label_path(subject_id, session)
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
        raw.pick(cls.eeg_channels)
        raw._data[np.isnan(raw._data)] = 0.0
        raw.resample(preprocessing_dict["sfreq"], verbose=False)

        l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
        if l_freq is not None or h_freq is not None:
            raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

        events, _ = mne.events_from_annotations(raw, event_id={"768": 768}, verbose=False)
        labels = loadmat(label_path)["classlabel"].reshape(-1).astype(np.int64) - 1
        if len(events) != len(labels):
            raise RuntimeError(
                f"Event/label count mismatch for subject {subject_id}, session {session}: "
                f"{len(events)} events vs {len(labels)} labels"
            )

        data = raw.get_data().astype(np.float32) * 1e6
        start_offset_samples, window_size = cls._window_spec(raw.info["sfreq"], preprocessing_dict)
        trials = []
        kept_labels = []
        for onset, label in zip(events[:, 0], labels):
            start_idx = onset + start_offset_samples
            stop_idx = start_idx + window_size
            segment = data[:, start_idx:stop_idx]
            if segment.shape == (cls.channels, window_size):
                trials.append(segment)
                kept_labels.append(label)

        if not trials:
            raise RuntimeError(f"No valid BCIC2b trials found in {gdf_path}")

        return np.stack(trials, axis=0).astype(np.float32), np.asarray(kept_labels, dtype=np.int64)

    def prepare_data(self) -> None:
        if self._has_local_official_subject_data(self.subject_id):
            train_parts = [
                self._load_session_from_official_files(self.subject_id, session, self.preprocessing_dict)
                for session in self.train_sessions
            ]
            test_parts = [
                self._load_session_from_official_files(self.subject_id, session, self.preprocessing_dict)
                for session in self.test_sessions
            ]
            self.dataset = {
                "train": (
                    np.concatenate([part[0] for part in train_parts], axis=0),
                    np.concatenate([part[1] for part in train_parts], axis=0),
                ),
                "test": (
                    np.concatenate([part[0] for part in test_parts], axis=0),
                    np.concatenate([part[1] for part in test_parts], axis=0),
                ),
            }
        else:
            self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2b",
                                      preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        if isinstance(self.dataset, dict):
            X, y = self.dataset["train"]
            X_test, y_test = self.dataset["test"]
        else:
            splitted_ds = self.dataset.split("session")
            train_datasets = [splitted_ds[f"session_{session}"] for session in [0, 1, 2]]
            test_datasets = [splitted_ds[f"session_{session}"] for session in [3, 4]]

            X = np.concatenate(
                [run.windows.load_data()._data for train_dataset in train_datasets for run
                 in train_dataset.datasets], axis=0)
            y = np.concatenate([run.y for train_dataset in train_datasets for run in
                                train_dataset.datasets], axis=0)
            X_test = np.concatenate(
                [run.windows.load_data()._data for test_dataset in test_datasets for run in
                test_dataset.datasets], axis=0)
            y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                     test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2bLOSO(BCICIV2b):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super(BCICIV2bLOSO, self).__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=self.all_subject_ids, dataset="2b",
            preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        train_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [0, 1, 2]]
        val_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [3, 4]]
        test_datasets = [
            splitted_ds[str(self.subject_id)].split("session")[f"session_{session}"]
            for session in [3, 4]]

        # load the data
        X = np.concatenate([run.windows.load_data()._data for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_val = np.concatenate([run.windows.load_data()._data for val_dataset in
                            val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                            val_dataset.datasets], axis=0)
        X_test = np.concatenate([run.windows.load_data()._data for test_dataset in test_datasets
                                 for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                 test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])
