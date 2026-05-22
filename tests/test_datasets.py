import json

import pytest

# These pull in schnetpack/matscipy/ase at construction time.
pytest.importorskip("schnetpack")
pytest.importorskip("matscipy")
ase_db = pytest.importorskip("ase.db")
from ase import Atoms  # noqa: E402

from koopmansvd.data.molecular import MolecularContextDataset  # noqa: E402


def _write_db(path, n_frames):
    """Write a tiny ASE DB with n_frames single-molecule rows (no .info.json)."""
    with ase_db.connect(str(path)) as conn:
        for i in range(n_frames):
            conn.write(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7 + 0.01 * i]]))


def test_len_without_info_json(tmp_path):
    """Regression: a DB without a .info.json sidecar must report a non-zero
    length of (raw_len - time_lag), not 0. A `self.len_` typo previously left
    __len__ at its initial 0, silently emptying the DataLoader."""
    n_frames = 6
    db_path = tmp_path / "traj.db"
    _write_db(db_path, n_frames)
    assert not db_path.with_suffix(".info.json").exists()

    ds1 = MolecularContextDataset(str(db_path), time_lag=1)
    assert len(ds1) == n_frames - 1  # window_span = 1 + time_lag

    ds3 = MolecularContextDataset(str(db_path), time_lag=3)
    assert len(ds3) == n_frames - 3


def test_len_with_info_json_respects_boundaries(tmp_path):
    """With a .info.json describing two trajectories, pair windows must not
    cross trajectory boundaries."""
    db_path = tmp_path / "traj.db"
    _write_db(db_path, 10)
    with open(db_path.with_suffix(".info.json"), "w") as f:
        json.dump({"traj_lengths": [5, 5]}, f)

    ds = MolecularContextDataset(str(db_path), time_lag=1)
    # Each length-5 trajectory yields (5 - 1) = 4 valid start indices.
    assert len(ds) == 8
