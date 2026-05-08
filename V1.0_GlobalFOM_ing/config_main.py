import torch
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config

def main():
    # data_dir = Path("./your_data_dir")   # 改成你的实际路径
    current_dir = Path(__file__).parent
    fig_dir = current_dir / "Fig"
    data_dir = current_dir / "Out_data"
    data_dir.mkdir(exist_ok=True)

    config = Config(device="cuda", dtype=torch.float64)
    config._create_uniform_grid_2D(data_dir)

    mesh_domain = torch.load(data_dir / "Mesh_domain.pt", weights_only=False)
    mesh_sub1   = torch.load(data_dir / "Mesh_subdomain_1.pt", weights_only=False)
    mesh_sub2   = torch.load(data_dir / "Mesh_subdomain_2.pt", weights_only=False)

    # ===================== 打印网格信息 =====================
    config.check_spacing(mesh_domain, "Domain")
    config.check_spacing(mesh_sub1, "Subdomain 1")
    config.check_spacing(mesh_sub2, "Subdomain 2")

    # ===================== 画图 =====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    config.plot_mesh(mesh_domain, axes[0], title="Full Domain Mesh")
    config.plot_mesh(mesh_sub1,   axes[1], title="Subdomain 1 Mesh")
    config.plot_mesh(mesh_sub2,   axes[2], title="Subdomain 2 Mesh")

    fig.tight_layout()
    fig.savefig(fig_dir / "Grid_domain.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()