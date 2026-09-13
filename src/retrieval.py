import numpy as np


def retrieve_closest(target_peaking_db, repository, max_error_threshold=1.0):
    """
    Finds the closest 'successful' past episode -- one where the achieved
    peaking_db actually landed close to what it was aiming for. Filtering
    to 'good' episodes matters: early-training episodes are close to random
    exploration and achieved something far from their target -- retrieving
    one of those would hand a new request a bad starting point.
    """
    good = [r for r in repository
            if abs(r['peaking_db'] - r['target_peaking_db']) < max_error_threshold]
    if not good:
        return None
    distances = [abs(r['peaking_db'] - target_peaking_db) for r in good]
    return good[int(np.argmin(distances))]


def retrieve_k_nearest(target_peaking_db, repository, k=10, max_error_threshold=1.0):
    """Same filtering as retrieve_closest, but returns the k best matches --
    needed for behavior-cloning pretraining, which works better with a
    small batch of examples than a single data point."""
    good = [r for r in repository
            if abs(r['peaking_db'] - r['target_peaking_db']) < max_error_threshold]
    if not good:
        return []
    good.sort(key=lambda r: abs(r['peaking_db'] - target_peaking_db))
    return good[:k]


def rs_cs_to_action(Rs, Cs, rs_range, cs_range):
    """Inverse of EqualizerEnv._rescale_action: converts real Rs/Cs values
    back into the normalized [-1, 1] action space the policy network
    operates in."""
    rs_low, rs_high = rs_range
    cs_low, cs_high = cs_range
    a0 = 2 * (Rs - rs_low) / (rs_high - rs_low) - 1
    a1 = 2 * (Cs - cs_low) / (cs_high - cs_low) - 1
    return np.array([a0, a1], dtype=np.float32)
