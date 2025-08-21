import numpy as np
import yaml


def spec2idx(i, j, nfields):
    if i == j:
        return i  # auto
    elif i < j:
        return nfields + (i * (2 * nfields - i - 1)) // 2 + (j - i - 1)
    else:
        return spec2idx(j, i, nfields)


def idx2spec(idx, nfields):
    if idx < nfields:
        return idx, idx
    idx_cross = idx - nfields
    total_cross = nfields * (nfields - 1) // 2
    if idx_cross < 0 or idx_cross >= total_cross:
        msg = f"Index {idx} out of bounds for nfields={nfields}"
        raise ValueError(msg)
    i = 0
    while idx_cross >= nfields - i - 1:
        idx_cross -= nfields - i - 1
        i += 1
    j = i + idx_cross + 1
    return i, j


class InputParams:
    def __init__(self):
        self.nside = 16
        self.spins = [0, 2]  # TQU
        self.labels = ["T", "E", "B"]

        self.feedback = 1
        self.inputclfile = "inputs/cls.dat"
        self.maskfile = "inputs/mask.fits"
        self.do_cross = True
        self.covmatfile1 = "inputs/NCVM1.bin"
        self.outinvcovmatfile1 = "outputs/invCOV1.bin"
        self.covmatfile2 = "inputs/NCVM2.bin"
        self.outinvcovmatfile2 = "outputs/invCOV2.bin"
        self.outnoisecovmat1 = "outputs/reducedNCVM1.bin"
        self.calibration = 1.0
        self.load_inverted = False
        self.output_geometry_file = "outputs/geometry.dat"
        self.smoothing_type = 2
        self.apply_pixwin = True
        self.smooth_pol = True
        self.fwhmarcmin = 440.0
        self.beam_file = "inputs/beam.fits"
        self.lmax = 64
        self.outfilefisher = "outputs/fisher.dat"
        self.ordering = 1

        self.nsims = None
        self.ssim = 1
        self.zerofill = 3
        self.endname1 = ""
        self.endname2 = ""
        self.inputmapfile1 = ""
        self.inputmapfile2 = ""
        self.outcovmatfile = ""
        self.outerrfile = ""
        self.remove_nb = True

        self.compute_derived()

    def compute_derived(self):
        self.nfields = len(self.labels)
        self.nspectra = self.nfields * (self.nfields + 1) // 2

        self.cross_idxs = np.array(
            [
                spec2idx(spec1, spec2, self.nfields)
                for spec1 in range(self.nfields)
                for spec2 in range(spec1 + 1, self.nfields)
            ]
        )
        self.auto_idxs = np.array(
            [spec2idx(spec, spec, self.nfields) for spec in range(self.nfields)]
        )

    def update(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.compute_derived()

    @staticmethod
    def read_parameter_file(yaml_file):
        with open(yaml_file) as file:
            config = yaml.safe_load(file)

        params = InputParams()
        params.update(config)
        return params

    def __str__(self):
        return "\n".join(
            f"{key}: {value}" for key, value in sorted(self.__dict__.items())
        )

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, InputParams):
            return False
        return all(
            getattr(self, key) == getattr(other, key) for key in self.__dict__.keys()
        )
