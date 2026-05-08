__all__ = ['device', 'dtype']

import json
import torch
from pathlib import Path
from fractions import Fraction
from config import Config
from ADR_solver import ADRSolver

current_dir = Path(__file__).parent
outdata_dir = current_dir / "Out_data"
outdata_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

def convert_fraction_strings(obj):
    if isinstance(obj, dict):
        return {k: convert_fraction_strings(v) for k, v in obj.items()}
    elif isinstance(obj, str) and '/' in obj and obj.replace('/', '').replace('-', '').isdigit():
        return Fraction(obj)
    else:
        return obj



def main():
    param_filename = current_dir / 'parameters.json'
    with open(param_filename, 'r') as f:
        phy_dict = json.load(f)
        phy_para = convert_fraction_strings(phy_dict)
        config = Config(device, dtype)
        # config._create_uniform_grid_2D(outdata_dir)

    adrsolver = ADRSolver(device, dtype, phy_para, outdata_dir)

    result = adrsolver._main_line()
    torch.save(result, outdata_dir / f'HF_result.pt')


if __name__ == "__main__":

    main()
