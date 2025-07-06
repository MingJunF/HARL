import logging
from typing import TYPE_CHECKING, Callable, Optional
import numpy as np
from collections import defaultdict

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils import vizard
if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


class RevisitImageData(Data):
    """Data for revisited images of targets (object version)."""

    def __init__(
        self,
        imaged_count: Optional[dict] = None,
        image_times: Optional[dict] = None,
        known: Optional[list["Target"]] = None,
    ) -> None:
        if imaged_count is None:
            imaged_count = {}
        self.imaged_count = imaged_count  # {target_obj: count}

        if image_times is None:
            image_times = defaultdict(list)
        self.image_times = image_times  # {target_obj: [timestamps]}

        if known is None:
            known = []
        self.known = list(set(known))

    def add_image(self, target, sim_time):
        if target in self.imaged_count:
            self.imaged_count[target] += 1
        else:
            self.imaged_count[target] = 1
        self.image_times[target].append(sim_time)

    def __add__(self, other: "RevisitImageData") -> "RevisitImageData":
        new_imaged_count = self.imaged_count.copy()
        new_image_times = defaultdict(list, {k: v[:] for k, v in self.image_times.items()})

        for target, count in other.imaged_count.items():
            new_imaged_count[target] = new_imaged_count.get(target, 0) + count

        for target, times in other.image_times.items():
            new_image_times[target].extend(times)

        known = list(set(self.known + other.known))
        return self.__class__(
            imaged_count=new_imaged_count, image_times=new_image_times, known=known
        )


class RevisitImageStore(DataStore):
    """DataStore for revisited images of targets."""

    data_type = RevisitImageData

    def get_log_state(self) -> np.ndarray:
        return np.array(
            self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
        )

    def compare_log_states(
        self, old_state: np.ndarray, new_state: np.ndarray
    ) -> RevisitImageData:
        update_idx = np.where(new_state - old_state > 0)[0]
        imaged = []
        for idx in update_idx:
            message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
            target_id = message.read().storedDataName[int(idx)]
            target = next((t for t in self.data.known if t.id == target_id), None)
            if target is not None:
                imaged.append((target, self.satellite.simulator.sim_time))

        data = RevisitImageData()
        for target, sim_time in imaged:
            data.add_image(target, sim_time)
        self.update_target_colors(imaged)
        return data

    @vizard.visualize
    def update_target_colors(self, targets, vizInstance=None, vizSupport=None):
        """Update target colors in Vizard."""
        for location in vizInstance.locations:
            if location.stationName in [target.name for (target, _) in targets]:
                location.color = vizSupport.toRGBA255(self.satellite.vizard_color)
class RevisitImageReward(GlobalReward):
    """GlobalReward for rewarding revisited images with round-dependent revisit constraints."""

    datastore_type = RevisitImageStore

    def __init__(
        self,
        reward_fn: Callable = lambda p: p,
        first_reward: float = 0.1,
        revisit_numrequest: int = 1,
    ) -> None:
        super().__init__()
        self.reward_fn = reward_fn
        self.first_reward = first_reward
        self.revisit_numrequest = revisit_numrequest
        self.completed_targets = set()
        self.Completeness = {}

    def initial_data(self, satellite: "Satellite") -> "RevisitImageData":
        return self.data_type(known=self.scenario.targets)

    def create_data_store(self, satellite: "Satellite") -> None:
        super().create_data_store(satellite)

        def revisit_target_filter(opportunity):
            if opportunity["type"] == "target":
                return opportunity["object"] not in self.completed_targets
            return True

        satellite.add_access_filter(revisit_target_filter)

    def calculate_reward(
        self, new_data_dict: dict[str, RevisitImageData]
    ) -> dict[str, float]:
        reward = {sat_id: 0.0 for sat_id in new_data_dict}
        all_observations = defaultdict(list)  # {target: [(sat_id, time)]}


        for sat_id, new_data in new_data_dict.items():
            for target, times in new_data.image_times.items():
                for t in times:
                    all_observations[target].append((sat_id, t))


        for target, times in self.data.image_times.items():
            for t in times:
                all_observations[target].append(("HIST", t))


        for target, observations in all_observations.items():
            if self.Completeness.get(target, 0.0) >= 1.0:
                continue

            observations.sort(key=lambda x: x[1])  

            first_sat = None
            first_time = None
            for sat_id, t in observations:
                if sat_id != "HIST":
                    reward[sat_id] += 0.1
                    first_sat = sat_id
                    first_time = t
                    break

            if first_sat is None:
                continue 
            for sat_id, t in observations:
                if sat_id != first_sat and sat_id != "HIST" and abs(t - first_time) <= 300:
                    reward[sat_id] += 0.9
                    self.Completeness[target] = 1.0
                    break

        return reward





