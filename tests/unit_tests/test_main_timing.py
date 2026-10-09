from types import SimpleNamespace

import numpy as np

from ffits.main import _write_timing_report
from ffits.ts_guess.guess import get_ts_guess, get_ts_guess_from_xyz


def test_write_timing_report_creates_file(tmp_path):
    report_file = tmp_path / "ffits_timing.log"

    _write_timing_report(
        timing_entries=[
            "xTB calculations: 0.010s",
            "FF parameterization: 0.020s",
            "TSFF construction: 0.030s",
            "Optimization: 1.234s",
        ],
        total_elapsed=1.244,
        success=True,
        output_path=report_file,
    )

    content = report_file.read_text()

    assert "FFiTS timing report" in content
    assert "xTB calculations: 0.010s" in content
    assert "FF parameterization: 0.020s" in content
    assert "TSFF construction: 0.030s" in content
    assert "Optimization: 1.234s" in content
    assert "Total time: 1.244s" in content
    assert "Status: COMPLETED" in content


def test_get_ts_guess_records_timing_phases(monkeypatch):
    entries = []

    def record_timing(timing_entries, timing_logger, label, start, end):
        timing_entries.append(label)

    def fake_create_tsff(**kwargs):
        return {"tsff": SimpleNamespace(start_from_reactant=True), "params_mix": {}}

    def fake_optimize_with_forcefield(*args, **kwargs):
        return True, 0.0, np.zeros((1, 3))

    monkeypatch.setattr("ffits.ts_guess.guess._record_timing", record_timing)
    monkeypatch.setattr("ffits.ts_guess.guess.create_tsff", fake_create_tsff)
    monkeypatch.setattr(
        "ffits.ts_guess.guess.optimize_with_forcefield", fake_optimize_with_forcefield
    )
    monkeypatch.setattr(
        "ffits.ts_guess.guess.print_ts_optimization_start", lambda: None
    )

    structure = SimpleNamespace(
        ff=SimpleNamespace(nat=1),
        info=SimpleNamespace(),
        path=SimpleNamespace(xyz_filename="input.xyz"),
    )

    get_ts_guess(
        structure,
        structure,
        timing_logger=SimpleNamespace(),
        timing_entries=entries,
    )

    assert entries == ["TSFF construction", "Optimization"]


def test_get_ts_guess_from_xyz_records_x_tb_and_parameterization(monkeypatch):
    entries = []

    def record_timing(timing_entries, timing_logger, label, start, end):
        timing_entries.append(label)

    def fake_structure_from_config(**kwargs):
        return SimpleNamespace(
            ff=SimpleNamespace(nat=1),
            info=SimpleNamespace(),
            path=SimpleNamespace(xyz_filename="input.xyz"),
        )

    def fake_parameterize_ff(*args, **kwargs):
        return None

    def fake_get_ts_guess(*args, **kwargs):
        return SimpleNamespace(start_from_reactant=True), True, 0.0, np.zeros((1, 3))

    monkeypatch.setattr("ffits.ts_guess.guess._record_timing", record_timing)
    monkeypatch.setattr("ffits.ts_guess.guess.Structure.from_config", fake_structure_from_config)
    monkeypatch.setattr("ffits.ts_guess.guess.parameterize_ff", fake_parameterize_ff)
    monkeypatch.setattr("ffits.ts_guess.guess.get_ts_guess", fake_get_ts_guess)
    monkeypatch.setattr("ffits.ts_guess.guess.print_calculation_data", lambda *args, **kwargs: None)
    monkeypatch.setattr("ffits.ts_guess.guess.print_header_setup", lambda *args, **kwargs: None)

    calcdata = SimpleNamespace(
        reactant_path=SimpleNamespace(xyz_filename=""),
        product_path=SimpleNamespace(xyz_filename=""),
        reactant_calc=SimpleNamespace(
            only_proper_dihedrals=True,
            ff_parameterization=True,
        ),
        product_calc=SimpleNamespace(
            only_proper_dihedrals=True,
            ff_parameterization=True,
        ),
    )

    get_ts_guess_from_xyz(
        "reac.xyz",
        "prod.xyz",
        calcdata=calcdata,
        timing_logger=SimpleNamespace(),
        timing_entries=entries,
    )

    assert entries == ["xTB calculations", "FF parameterization"]
