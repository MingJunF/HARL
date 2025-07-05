"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.matd3 import MATD3
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.mappo_potential import MAPPO_Potential
from harl.algorithms.actors.r_mappo import MAPPO_Potential_V2
from harl.algorithms.actors.happo_potential import HAPPO_Potential
from harl.algorithms.actors.hatrpo_potential import HATRPO_Potential

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
    "mappo_potential_v2": MAPPO_Potential_V2,
    "mappo_potential": MAPPO_Potential,
    "happo_potential": HAPPO_Potential,
    "hatrpo_potential": HATRPO_Potential
}
