from TransGAT.configs.prepareGEF import PrepareGEF
from TransGAT.configs.prepareWPF import PrepareWPF
config_dict = {
    "WPF": PrepareWPF,
    "GEF": PrepareGEF
}


def config_factory(data):
    args = config_dict[data]()()
    return args
