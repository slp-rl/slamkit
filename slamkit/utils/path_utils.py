import os

def resolve_reference_path(path, default_path):
    if path.startswith('//reference'):
        reference_path = os.environ.get("SLAM_REFERENCE_PATH", default_path)
        assert reference_path is not None, "SLAM_REFERENCE_PATH is not set and not set in config"
        path = path.replace('//reference', reference_path)
    return path