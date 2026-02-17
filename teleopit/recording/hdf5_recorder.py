"""HDF5-based data recorder for teleoperation sessions."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


class HDF5Recorder:
    """Records teleoperation data to HDF5 format with chunked storage and compression.
    
    Supports numerical data fields: joint_pos, joint_vel, mimic_obs, action, timestamp.
    First call to add_frame creates datasets based on data keys/shapes (resizable).
    Subsequent calls resize and append data.
    """
    
    def __init__(self, path: str | Path, chunk_size: int = 100):
        """Initialize HDF5 recorder.
        
        Args:
            path: Path to HDF5 file to create
            chunk_size: Chunk size for HDF5 datasets (affects compression/performance)
        """
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.file: h5py.File | None = None
        self.datasets: Dict[str, h5py.Dataset] = {}
        self.frame_count = 0
        self.start_time = time.time()
        self._initialized = False
        
    def __enter__(self) -> HDF5Recorder:
        """Context manager entry."""
        self.file = h5py.File(self.path, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
        
    def add_frame(self, data: Dict[str, np.ndarray]) -> None:
        """Add a frame of data to the recording.
        
        First call creates datasets based on data keys/shapes.
        Subsequent calls resize and append data.
        
        Args:
            data: Dictionary mapping field names to numpy arrays
                  Supported fields: joint_pos, joint_vel, mimic_obs, action, timestamp
        """
        if self.file is None:
            raise RuntimeError("Recorder not opened. Use context manager or call __enter__")
            
        if not self._initialized:
            self._create_datasets(data)
            self._initialized = True
            
        # Resize and write to each dataset
        for key, value in data.items():
            if key not in self.datasets:
                raise ValueError(f"Unexpected key '{key}' not in initial frame")
                
            dataset = self.datasets[key]
            # Resize to accommodate new frame
            dataset.resize(self.frame_count + 1, axis=0)
            # Write data
            dataset[self.frame_count] = value
            
        self.frame_count += 1
        
    def _create_datasets(self, data: Dict[str, np.ndarray]) -> None:
        """Create resizable HDF5 datasets based on first frame data.
        
        Args:
            data: First frame of data defining dataset structure
        """
        for key, value in data.items():
            arr = np.asarray(value)
            shape = arr.shape
            dtype = arr.dtype
            
            # Create resizable dataset with chunking and compression
            chunk_shape = (self.chunk_size,) + shape
            max_shape = (None,) + shape  # Unlimited first dimension
            
            self.datasets[key] = self.file.create_dataset(
                key,
                shape=(0,) + shape,
                maxshape=max_shape,
                chunks=chunk_shape,
                dtype=dtype,
                compression='gzip',
                compression_opts=4
            )
            
    def close(self) -> None:
        """Close the recorder and write metadata."""
        if self.file is None:
            return
            
        # Write metadata
        end_time = time.time()
        recording_duration = end_time - self.start_time
        
        self.file.attrs['total_frames'] = self.frame_count
        self.file.attrs['recording_time'] = recording_duration
        self.file.attrs['start_time'] = self.start_time
        self.file.attrs['end_time'] = end_time
        
        self.file.close()
        self.file = None
