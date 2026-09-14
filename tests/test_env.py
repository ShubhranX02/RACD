"""
test_env.py — Tests for environment_transistor.py and environment_goal.py

Tests:
  - TransistorEqualizerEnv obs space is 9-D
  - reset() returns 9-D obs
  - obs values are in [0, 1] (normalised)
  - corner one-hot is valid (exactly one 1 in positions 4-8)
  - peaking_target_range default is (3.0, 12.0)
  - environment_goal.py goal_high[2] >= 3000
  - validate_pvt PASS criteria includes peaking
"""
import numpy as np
import pytest


def test_obs_space_is_9d():
    """TransistorEqualizerEnv observation_space must be Box(9,)."""
    from environment_transistor import TransistorEqualizerEnv
    env = TransistorEqualizerEnv(use_surrogate=False, multi_corner=False)
    assert env.observation_space.shape == (9,), (
        f"Expected obs shape (9,), got {env.observation_space.shape}"
    )


def test_reset_returns_9d_obs():
    """reset() must return a (9,) numpy array."""
    from environment_transistor import TransistorEqualizerEnv
    env = TransistorEqualizerEnv(use_surrogate=False, multi_corner=False)
    obs, info = env.reset(seed=42)
    assert obs.shape == (9,), f"Expected obs shape (9,), got {obs.shape}"
    assert obs.dtype == np.float32


def test_obs_values_in_unit_range():
    """All obs values should be in [0, 1] (normalised)."""
    from environment_transistor import TransistorEqualizerEnv
    env = TransistorEqualizerEnv(use_surrogate=False, multi_corner=False)
    for _ in range(20):
        obs, _ = env.reset()
        assert np.all(obs >= -0.01), f"Obs has values below 0: {obs}"
        assert np.all(obs <= 1.01), f"Obs has values above 1: {obs}"


def test_corner_onehot_valid():
    """Positions 4-8 of obs must be a valid one-hot (exactly one 1)."""
    from environment_transistor import TransistorEqualizerEnv
    env = TransistorEqualizerEnv(use_surrogate=False, multi_corner=True)
    corners_seen = set()
    for _ in range(30):
        obs, _ = env.reset()
        corner_bits = obs[4:9]
        assert corner_bits.sum() == pytest.approx(1.0, abs=1e-5), (
            f"Corner one-hot should sum to 1, got {corner_bits}"
        )
        corners_seen.add(int(np.argmax(corner_bits)))
    # With multi_corner=True and 30 episodes, we should see >1 corner
    assert len(corners_seen) > 1, "multi_corner=True should randomize corners"


def test_peaking_target_range_default():
    """Default peaking_target_range must be (3.0, 12.0) to cover the full spec."""
    from environment_transistor import TransistorEqualizerEnv
    import inspect
    sig = inspect.signature(TransistorEqualizerEnv.__init__)
    default = sig.parameters['peaking_target_range'].default
    assert default == (3.0, 12.0), f"Expected (3.0, 12.0), got {default}"


def test_goal_env_eye_clipping_fixed():
    """environment_goal.py goal_high[2] must be >= 3000 to avoid clipping."""
    from environment_goal import GoalEqualizerEnv
    env = GoalEqualizerEnv()
    # The goal space upper bound for eye height proxy (index 2)
    # It is in the desired_goal / achieved_goal space
    desired_goal_high = env.observation_space['desired_goal'].high
    assert desired_goal_high[2] >= 3000.0, (
        f"goal_high[2] should be >= 3000 mV to avoid clipping, got {desired_goal_high[2]}"
    )


def test_validate_pvt_pass_requires_peaking_accuracy():
    """validate_pvt PASS criteria must include peaking accuracy check."""
    import inspect
    from validate_pvt import run_pvt_validation
    # Read the source to verify peaking_ok variable exists
    import validate_pvt
    src = inspect.getsource(validate_pvt)
    assert 'peaking_ok' in src, "validate_pvt must contain peaking_ok variable for PASS criteria"
    assert 'target_peaking_db' in src
