from src.fedalg.fedavg import FedAvg, FedSgd
from src.fedalg.fedcdp import FedCdp
from src.fedalg.fedavg_snn import FedAvg_SNN

fedlearning_registry = {
    "fedsgd":   FedSgd,
    "fedcdp":   FedCdp, 
    "fedavg":   FedAvg,

    "fedavg_snn": FedAvg_SNN,
}