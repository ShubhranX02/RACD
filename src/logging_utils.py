import json
import os


def log_episode(info, log_path=None):
    """
    Appends one episode as a single line of JSON (JSON Lines format).
    Cost per call is constant regardless of repository size -- no reading
    or rewriting of existing data, unlike a single growing JSON array.
    """
    if log_path is None:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'circuit_repository.jsonl')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(info) + '\n')


def load_repository(log_path=None):
    if log_path is None:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'circuit_repository.jsonl')
    records = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records
