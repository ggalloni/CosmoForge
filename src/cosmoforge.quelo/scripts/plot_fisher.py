import numpy as np
from matplotlib import pyplot as plt


def fisher_plotting(show_fig=False, save_fig=False):
    print("Hello from test!")

    file = "src/cosmoforge.quelo/outputs/TEB_fisher.dat"
    # file = "src/cosmoforge.quelo/inputs/EB_ns008_1ch_r0.0_fsky100_AxA_QU_fisher.dat"
    fisher = np.loadtxt(file, dtype=np.float64)  # Use double precision

    n_ell = 32 - 1

    # idx = 3
    # EB_block = fisher[n_ell * idx : n_ell * (idx + 1), n_ell * idx : n_ell * (idx + 1)]

    # plt.imshow(EB_block, origin="lower")
    # plt.colorbar()
    # plt.title("EB Block of Fisher Matrix")
    # plt.show()

    # idx = 4
    # TE_block = fisher[n_ell * idx : n_ell * (idx + 1), n_ell * idx : n_ell * (idx + 1)]

    # idx = 5
    # TB_block = fisher[n_ell * idx : n_ell * (idx + 1), n_ell * idx : n_ell * (idx + 1)]

    # idx = 3
    # fisher[n_ell * idx : n_ell * (idx + 1), n_ell * idx : n_ell * (idx + 1)] = (
    #     TE_block.copy()
    # )

    # idx = 4
    # fisher[n_ell * idx : n_ell * (idx + 1), n_ell * idx : n_ell * (idx + 1)] = (
    #     TB_block.copy()
    # )

    # idx = 5
    # fisher[n_ell * idx : n_ell * (idx + 1), n_ell * idx : n_ell * (idx + 1)] = (
    #     EB_block.copy()
    # )

    # file = "src/cosmoforge.quelo/tests/data/QU_ref_fisher.dat"
    file = "src/cosmoforge.quelo/tests/data/TEB_ref_fisher.dat"
    # file = "src/cosmoforge.quelo/tests/data/EB_ref_fisher.dat"
    fisher2 = np.loadtxt(file, dtype=np.float64)  # Use double precision

    # # Compute relative and absolute differences
    # diff = fisher - fisher2
    # rel_diff = np.abs(diff) / (np.abs(fisher2) + 1e-15)  # Avoid division by zero

    # print(f"Fisher matrix shape: {fisher.shape}")
    # print(f"Max absolute difference: {np.max(np.abs(diff)):.2e}")
    # print(f"Mean absolute difference: {np.mean(np.abs(diff)):.2e}")
    # print(f"Max relative difference: {np.max(rel_diff):.2e}")
    # print(f"Mean relative difference: {np.mean(rel_diff):.2e}")
    # print(
    #     f"Typical Fisher values: min={np.min(np.abs(fisher2)):.2e}, "
    #     f"max={np.max(np.abs(fisher2)):.2e}"
    # )

    # # Check if differences are mostly on diagonal vs off-diagonal
    # diag_mask = np.eye(fisher.shape[0], dtype=bool)
    # print(f"Diagonal max rel diff: {np.max(rel_diff[diag_mask]):.2e}")
    # print(f"Off-diagonal max rel diff: {np.max(rel_diff[~diag_mask]):.2e}")

    # plt.figure(figsize=(8, 6))
    # plt.imshow(
    #     fisher,
    #     origin="upper",
    # )
    # plt.colorbar()
    # plt.title("Fisher Matrix QUELO")

    # plt.figure(figsize=(8, 6))
    # plt.imshow(
    #     fisher2,
    #     origin="upper",
    # )
    # plt.colorbar()
    # plt.title("Fisher Matrix pse_qml")

    # # diff = fisher - fisher2
    # # plt.figure(figsize=(8, 6))
    # # plt.imshow(
    # #     diff,
    # #     origin="upper",
    # # )
    # # plt.colorbar()
    # # plt.title("Absolute Difference")

    # plt.show()

    couples = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    ]
    for i, j in couples:
        print(f"Processing block ({i}, {j})")

        plt.figure(figsize=(8, 6))
        plt.imshow(
            fisher[n_ell * i : n_ell * (i + 1), n_ell * i : n_ell * (i + 1)],
            origin="upper",
        )
        plt.colorbar()
        plt.title(f"Fisher Matrix QUELO - Block ({i})")
        if save_fig:
            plt.savefig(
                f"src/cosmoforge.quelo/plots/QUELO_fisher_block_{i}.png",
                bbox_inches="tight",
            )

        plt.figure(figsize=(8, 6))
        plt.imshow(
            fisher2[n_ell * j : n_ell * (j + 1), n_ell * j : n_ell * (j + 1)],
            origin="upper",
        )
        plt.colorbar()
        plt.title(f"Fisher Matrix pse_qml - Block ({j})")
        if save_fig:
            plt.savefig(
                f"src/cosmoforge.quelo/plots/pse_qml_fisher_block_{j}.png",
                bbox_inches="tight",
            )

        diff = (
            fisher[n_ell * i : n_ell * (i + 1), n_ell * i : n_ell * (i + 1)]
            - fisher2[n_ell * j : n_ell * (j + 1), n_ell * j : n_ell * (j + 1)]
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(
            diff,
            origin="upper",
        )
        plt.colorbar()
        plt.title(f"Absolute Difference - Block ({i}, {j})")
        if save_fig:
            plt.savefig(
                f"src/cosmoforge.quelo/plots/QUELO_pse_qml_fisher_diff_block_{i}_{j}.png",
                bbox_inches="tight",
            )
        if show_fig:
            plt.show()

    # off diagonal blocks

    plt.figure(figsize=(8, 6))
    plt.imshow(
        fisher[:n_ell, n_ell : n_ell * 3],
        origin="upper",
    )
    plt.colorbar()
    plt.title("Fisher Matrix QUELO - Off-diagonal Block (T-QU)")
    if save_fig:
        plt.savefig(
            "src/cosmoforge.quelo/plots/QUELO_fisher_offdiag_block_T-QU.png",
            bbox_inches="tight",
        )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        fisher2[:n_ell, n_ell : n_ell * 3],
        origin="upper",
    )
    plt.colorbar()
    plt.title("Fisher Matrix pse_qml - Off-diagonal Block (T-QU)")
    if save_fig:
        plt.savefig(
            "src/cosmoforge.quelo/plots/pse_qml_fisher_offdiag_block_T-QU.png",
            bbox_inches="tight",
        )

    diff = fisher[:n_ell, n_ell : n_ell * 3] - fisher2[:n_ell, n_ell : n_ell * 3]
    plt.figure(figsize=(8, 6))
    plt.imshow(
        diff,
        origin="upper",
    )
    plt.colorbar()
    plt.title("Absolute Difference - Off-diagonal Block (T-QU)")
    if save_fig:
        plt.savefig(
            "src/cosmoforge.quelo/plots/QUELO_pse_qml_fisher_offdiag_diff_block_T-QU.png",
            bbox_inches="tight",
        )
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(
        fisher,
        origin="upper",
    )
    plt.colorbar()
    plt.title("Fisher Matrix QUELO - Off-diagonal Block full")
    if save_fig:
        plt.savefig(
            "src/cosmoforge.quelo/plots/QUELO_fisher_full.png", bbox_inches="tight"
        )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        fisher2,
        origin="upper",
    )
    plt.colorbar()
    plt.title("Fisher Matrix pse_qml - Off-diagonal Block full")
    if save_fig:
        plt.savefig(
            "src/cosmoforge.quelo/plots/pse_qml_fisher_full.png", bbox_inches="tight"
        )

    diff = fisher - fisher2
    plt.figure(figsize=(8, 6))
    plt.imshow(
        diff,
        origin="upper",
    )
    plt.colorbar()
    plt.title("Absolute Difference - Off-diagonal Block full")
    if save_fig:
        plt.savefig(
            "src/cosmoforge.quelo/plots/QUELO_pse_qml_fisher_offdifull.png",
            bbox_inches="tight",
        )
    if show_fig:
        plt.show()


if __name__ == "__main__":
    fisher_plotting(show_fig=True, save_fig=False)
