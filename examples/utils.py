import sys
import os
import argparse

def add_sher_to_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.append(parent_dir)

    from sher import Lion, Betas2, Truncated, PI
    return Lion, Betas2, Truncated, PI

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)

def truncated_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    parts = strings.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Truncated must be a tuple of (bool, float or None)")
    bool_part = parts[0].strip().lower() == 'true'
    if parts[1].strip().lower() == 'none':
        return (bool_part, None)
    else:
        return (bool_part, float(parts[1]))
        
def update_config_from_args(config, args):
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))