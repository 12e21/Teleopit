"""Tests for teleopit.recording.hdf5_recorder — HDF5 write/read, context manager, empty file."""
import numpy as np
import pytest

from conftest import requires_h5py


@requires_h5py
class TestHDF5RecorderContextManager:
    """Context manager and basic write/read."""

    def test_context_manager_creates_file(self, tmp_dir):
        from teleopit.recording.hdf5_recorder import HDF5Recorder

        path = tmp_dir / "test.h5"
        with HDF5Recorder(path) as rec:
            assert rec.file is not None
        # After exit, file should be closed
        assert rec.file is None
        assert path.exists()

    def test_write_and_read_frames(self, tmp_dir):
        import h5py
        from teleopit.recording.hdf5_recorder import HDF5Recorder

        path = tmp_dir / "test.h5"
        n_frames = 5
        joint_dim = 29

        with HDF5Recorder(path) as rec:
            for i in range(n_frames):
                rec.add_frame({
                    "joint_pos": np.full(joint_dim, float(i), dtype=np.float32),
                    "timestamp": np.array(float(i)),
                })
            assert rec.frame_count == n_frames

        # Read back
        with h5py.File(path, "r") as f:
            assert f["joint_pos"].shape == (n_frames, joint_dim)
            assert f["timestamp"].shape == (n_frames,)
            assert f.attrs["total_frames"] == n_frames
            np.testing.assert_allclose(f["joint_pos"][0], 0.0)
            np.testing.assert_allclose(f["joint_pos"][4], 4.0)

    def test_empty_file_has_metadata(self, tmp_dir):
        import h5py
        from teleopit.recording.hdf5_recorder import HDF5Recorder

        path = tmp_dir / "empty.h5"
        with HDF5Recorder(path) as rec:
            pass  # no frames added

        with h5py.File(path, "r") as f:
            assert f.attrs["total_frames"] == 0
            assert "recording_time" in f.attrs


@requires_h5py
class TestHDF5RecorderErrors:
    """Error handling."""

    def test_add_frame_without_context_raises(self, tmp_dir):
        from teleopit.recording.hdf5_recorder import HDF5Recorder

        rec = HDF5Recorder(tmp_dir / "test.h5")
        with pytest.raises(RuntimeError, match="not opened"):
            rec.add_frame({"x": np.array([1.0])})

    def test_unexpected_key_raises(self, tmp_dir):
        from teleopit.recording.hdf5_recorder import HDF5Recorder

        path = tmp_dir / "test.h5"
        with HDF5Recorder(path) as rec:
            rec.add_frame({"a": np.array([1.0])})
            with pytest.raises(ValueError, match="Unexpected key"):
                rec.add_frame({"b": np.array([2.0])})

    def test_close_idempotent(self, tmp_dir):
        from teleopit.recording.hdf5_recorder import HDF5Recorder

        rec = HDF5Recorder(tmp_dir / "test.h5")
        rec.close()  # should not raise even if never opened
        rec.close()  # double close is fine
