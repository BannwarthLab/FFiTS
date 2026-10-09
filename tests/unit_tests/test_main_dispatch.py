"""Tests for ffits.main.main(): argument-dispatch logic and the
success/failure run-summary wrapping around it.

These tests mock parse_args() and all four run_*_mode() functions so they
exercise only the dispatch/try-except logic in main(), not the underlying
(heavy, xtb/molbar-dependent) calculations -- those are covered by the
tmp_workdir-based integration tests in tests/unit_tests/test_main.py and
tests/integration_tests/.
"""

import pytest

import ffits.main as main_mod


def _base_args(**overrides):
    args = {
        "structures": ["reac.xyz", "prod.xyz"],
        "config_file": None,
        "charge": None,
        "multiplicity": None,
        "parameterize": False,
        "reaction_path": None,
        "optff": None,
        "debug": False,
    }
    args.update(overrides)
    return args


@pytest.fixture(autouse=True)
def _quiet_setup(monkeypatch):
    """main() also calls print_program_header/setup_logger/overwrite_from_commandline
    and builds a CalculationData -- none of that is under test here, so no-op it."""
    monkeypatch.setattr(main_mod, "print_program_header", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "setup_logger", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "overwrite_from_commandline", lambda *a, **k: None)
    monkeypatch.setattr(
        main_mod.CalculationData, "from_default", staticmethod(lambda: object())
    )


def _track(monkeypatch, name):
    calls = []
    monkeypatch.setattr(
        main_mod, name, lambda *a, **k: calls.append((a, k)) or "result"
    )
    return calls


def test_dispatches_to_optimizer_mode(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: _base_args(optff="ff.csv", structures=["reac.xyz"]),
    )
    opt_calls = _track(monkeypatch, "run_optimizer_mode")
    param_calls = _track(monkeypatch, "run_parameterization_mode")
    rct_calls = _track(monkeypatch, "run_reaction_path_mode")
    ts_calls = _track(monkeypatch, "run_tsguess_mode")
    monkeypatch.setattr(main_mod, "print_run_summary", lambda *a, **k: None)

    main_mod.main()

    assert len(opt_calls) == 1
    assert opt_calls[0][0][:2] == ("reac.xyz", "ff.csv")
    assert param_calls == []
    assert rct_calls == []
    assert ts_calls == []


def test_dispatches_to_parameterization_mode(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: _base_args(parameterize=True, structures=["reac.xyz"]),
    )
    opt_calls = _track(monkeypatch, "run_optimizer_mode")
    param_calls = _track(monkeypatch, "run_parameterization_mode")
    rct_calls = _track(monkeypatch, "run_reaction_path_mode")
    ts_calls = _track(monkeypatch, "run_tsguess_mode")
    monkeypatch.setattr(main_mod, "print_run_summary", lambda *a, **k: None)

    main_mod.main()

    assert len(param_calls) == 1
    assert param_calls[0][0][0] == "reac.xyz"
    assert opt_calls == []
    assert rct_calls == []
    assert ts_calls == []


def test_dispatches_to_reaction_path_mode(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: _base_args(reaction_path=5))
    opt_calls = _track(monkeypatch, "run_optimizer_mode")
    param_calls = _track(monkeypatch, "run_parameterization_mode")
    rct_calls = _track(monkeypatch, "run_reaction_path_mode")
    ts_calls = _track(monkeypatch, "run_tsguess_mode")
    monkeypatch.setattr(main_mod, "print_run_summary", lambda *a, **k: None)

    main_mod.main()

    assert len(rct_calls) == 1
    assert rct_calls[0][1]["reactant_filename"] == "reac.xyz"
    assert rct_calls[0][1]["product_filename"] == "prod.xyz"
    assert rct_calls[0][1]["steps"] == 5
    assert opt_calls == []
    assert param_calls == []
    assert ts_calls == []


def test_dispatches_to_tsguess_mode_by_default(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: _base_args())
    opt_calls = _track(monkeypatch, "run_optimizer_mode")
    param_calls = _track(monkeypatch, "run_parameterization_mode")
    rct_calls = _track(monkeypatch, "run_reaction_path_mode")
    ts_calls = _track(monkeypatch, "run_tsguess_mode")
    monkeypatch.setattr(main_mod, "print_run_summary", lambda *a, **k: None)

    main_mod.main()

    assert len(ts_calls) == 1
    assert ts_calls[0][1]["reactant_filename"] == "reac.xyz"
    assert ts_calls[0][1]["product_filename"] == "prod.xyz"
    assert opt_calls == []
    assert param_calls == []
    assert rct_calls == []


def test_prints_success_summary_when_run_mode_succeeds(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: _base_args())
    monkeypatch.setattr(main_mod, "run_tsguess_mode", lambda *a, **k: None)
    summary_calls = []
    monkeypatch.setattr(
        main_mod,
        "print_run_summary",
        lambda *a, **k: summary_calls.append((a, k)),
    )

    main_mod.main()

    assert len(summary_calls) == 1
    args, kwargs = summary_calls[0]
    assert kwargs.get("success", True) is True


def test_prints_failure_summary_and_reraises_when_run_mode_fails(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: _base_args())

    def boom(*a, **k):
        raise ValueError("something went wrong")

    monkeypatch.setattr(main_mod, "run_tsguess_mode", boom)
    summary_calls = []
    monkeypatch.setattr(
        main_mod,
        "print_run_summary",
        lambda *a, **k: summary_calls.append((a, k)),
    )

    with pytest.raises(ValueError, match="something went wrong"):
        main_mod.main()

    assert len(summary_calls) == 1
    _, kwargs = summary_calls[0]
    assert kwargs["success"] is False
    assert kwargs["message"] == "something went wrong"


def test_debug_flag_sets_debug_log_level(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: _base_args(debug=True))
    monkeypatch.setattr(main_mod, "run_tsguess_mode", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "print_run_summary", lambda *a, **k: None)
    levels = []
    monkeypatch.setattr(main_mod, "setup_logger", lambda level: levels.append(level))

    main_mod.main()

    assert levels == ["DEBUG"]


def test_no_debug_flag_sets_info_log_level(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: _base_args(debug=False))
    monkeypatch.setattr(main_mod, "run_tsguess_mode", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "print_run_summary", lambda *a, **k: None)
    levels = []
    monkeypatch.setattr(main_mod, "setup_logger", lambda level: levels.append(level))

    main_mod.main()

    assert levels == ["INFO"]
