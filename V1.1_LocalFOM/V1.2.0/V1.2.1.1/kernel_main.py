__all__ = ['device', 'dtype']

import torch
from kernel_sovle import ADRSolver
# from kernel_sovle_implicit_interface import ADRSolver
# from kernel_sovle_implicit_dualtime import ADRSolver
from pathlib import Path

current_dir = Path(__file__).parent
outdata_dir = current_dir / "Out_data"
outdata_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)


def main():
    para_file = current_dir / 'subdomain_A_monolithic.json'
    adrsolver = ADRSolver(device, outdata_dir, para_file)
    u = adrsolver._main_line()




if __name__ == "__main__":

    main()
