"""Target scenarios distribute ground targets with some distribution.

Currently, targets are all known to the satellites a priori and are available based on
the imaging requirements given by the dynamics and flight software models.
"""

import logging
import os
import sys
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union
import random
import numpy as np
import pandas as pd
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import macros
from Basilisk.utilities.orbitalMotion import ClassicElements, elem2rv, rv2elem
from bsk_rl.scene import Scenario
from bsk_rl.utils.orbital import lla2ecef
from bsk_rl.utils.orbital import TrajectorySimulator
if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite
from bsk_rl.utils import vizard
logger = logging.getLogger(__name__)


class Target:
    """Ground target with associated value."""

    def __init__(self, name: str, r_LP_P: Iterable[float], priority: float, Round: int) -> None:
        """Ground target with associated priority and location.

        Args:
            name: Identifier; does not need to be unique
            r_LP_P: Planet-fixed, planet relative location [m]
            priority: Value metric.
        """
        self.name = name
        self.r_LP_P = np.array(r_LP_P)
        self.priority = priority
        self.Round = Round
    @property
    def id(self) -> str:
        """Get unique, human-readable identifier."""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __hash__(self) -> int:
        """Hash target by unique id."""
        return hash((self.id))

    def __repr__(self) -> str:
        """Get string representation of target.

        Use ``target.id`` for a unique string identifier.

        Returns:
            Target string
        """
        return f"Target({self.name})"

def find_targets_folder(start_path: str = __file__) -> Path:
    path = Path(start_path).resolve()
    for parent in [path] + list(path.parents):
        candidate = parent / "Targets"
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find 'Targets' folder from any parent directory.")


class Aushotspots(Scenario):
    """Environment with static targets loaded from CSV."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        target_location: str,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3
    ) -> None:
        self.target_location = target_location
        self.radius = radius
        self.targets = []
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()
        self.priority_distribution = priority_distribution
        self._n_targets = n_targets  

    def _load_targets_from_csv(self):
        base_path = find_targets_folder(__file__) 
        csv_path = base_path / f"{self.target_location}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Target file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            lat, lon = row["lat"], row["lon"]
            location = lla2ecef(lat, lon, self.radius)
            location /= np.linalg.norm(location)
            location *= self.radius
            target_name = row.get("user_request", f"Target_{idx}")
            round_value = int(row["round"]) 

            self.targets.append(
                Target(
                    name=target_name,
                    r_LP_P=location,
                    priority=self.priority_distribution(),
                    Round=round_value 
                )
            )

        self._n_targets = len(self.targets)


    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        logger.info(f"Loading targets from {self.target_location}.csv")
        
        self._load_targets_from_csv()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator["imageTargetMinimumElevation"],
                        type="target",
                    )
    def reset_during_sim_init(self) -> None:
        """Visualize targets in Vizard on reset."""
        for target in self.targets:
            self.visualize_target(target)

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,  # meters
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1
class Scenario2_2(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def regenerate_targets(self) -> None:

        self.targets = []
        satellite = self.satellites[0]
        mu = satellite.sat_args_generator["mu"]  

        num_clusters = 7      
        targets_per_cluster = 5  
        cluster_radius = 300000  


        for task in range(num_clusters):

            f = task * (2 * np.pi / num_clusters)

            cluster_orbit_params = ClassicElements()
            cluster_orbit_params.a = 6771000.0  
            cluster_orbit_params.e = 0          
            cluster_orbit_params.i = 0          
            cluster_orbit_params.Omega = 0      
            cluster_orbit_params.omega = 0      
            cluster_orbit_params.f = f        

            cluster_center_pos, _ = elem2rv(mu, cluster_orbit_params)
            #print(cluster_center_pos)
            cluster_center_pos = np.array(cluster_center_pos)

            for target_index in range(targets_per_cluster):
 
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, cluster_radius)
                x_offset = distance * np.cos(angle)
                y_offset = distance * np.sin(angle)
                z_offset = np.random.uniform(-300000, 300000)

                target_pos = cluster_center_pos + np.array([x_offset, y_offset, z_offset])
                target_pos *= self.radius / np.linalg.norm(target_pos)
  
                self.targets.append(
                    Target(
                        name=f"cluster_{task}_target_{target_index}",
                        r_LP_P=target_pos,
                        priority=1,
                        #self.priority_distribution(),
                        Round=task  
                    )
                )
            self.targets.append(
                Target(
                    name="Nadir",
                    r_LP_P=np.array([0,0.1,0]),
                    priority=0,
                    Round=0
                )
            )


class Scenario2_1(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def regenerate_targets(self) -> None:
        self.targets = []
        df = pd.read_csv("data/Scenario2.csv")

        for _, row in df.iterrows():
            self.targets.append(
                Target(
                    name=row["name"],
                    r_LP_P=np.array([row["x"], row["y"], row["z"]]),
                    priority=row["priority"],
                    Round=row["Round"]
                )
            )
        self.targets.append(
            Target(
                name="Nadir",
                r_LP_P=np.array([0,0.1,0]),
                priority=0,
                Round=0
            )
        )
class DenseTarget(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        cluster_radius: float = 300000.0,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius
        self.cluster_radius=cluster_radius
    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def regenerate_targets(self) -> None:


        self.targets = []
        satellite = self.satellites[0]
        mu = satellite.sat_args_generator["mu"] 
        
        num_clusters = int(self.n_targets/7)      
        targets_per_cluster = 7  
        cluster_radius =  self.cluster_radius   


        for task in range(num_clusters):

            f = task * (2 * np.pi / num_clusters)

            cluster_orbit_params = ClassicElements()
            cluster_orbit_params.a = 6771000.0  
            cluster_orbit_params.e = 0          
            cluster_orbit_params.i = 0          
            cluster_orbit_params.Omega = 0     
            cluster_orbit_params.omega = 0     
            cluster_orbit_params.f = f        


            cluster_center_pos, _ = elem2rv(mu, cluster_orbit_params)
 
            cluster_center_pos = np.array(cluster_center_pos)  

            for target_index in range(targets_per_cluster):

                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, cluster_radius)
                x_offset = distance * np.cos(angle)
                y_offset = distance * np.sin(angle)
                z_offset = np.random.uniform(-cluster_radius, cluster_radius) 


                target_pos = cluster_center_pos + np.array([x_offset, y_offset, z_offset])
                target_pos *= self.radius / np.linalg.norm(target_pos)
 
                self.targets.append(
                    Target(
                        name=f"cluster_{task}_target_{target_index}",
                        r_LP_P=target_pos,
                        priority=1,
                        Round=task 
                    )
                )
            self.targets.append(
                Target(
                    name="Nadir",
                    r_LP_P=np.array([0,0.1,0]),
                    priority=0,
                    Round=0
                )
            )
    def reset_during_sim_init(self) -> None:
        """Visualize targets in Vizard on reset."""
        for target in self.targets:
            self.visualize_target(target)

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("White"),
            range=1000.0 * 1000,  # meters
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1
class Scenario1_1(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )
    def regenerate_targets(self) -> None:
        self.targets = []
        df = pd.read_csv("data/Scenario1.csv")

        for _, row in df.iterrows():
            self.targets.append(
                Target(
                    name=row["name"],
                    r_LP_P=np.array([row["x"], row["y"], row["z"]]),
                    priority=row["priority"],
                    Round=row["Round"]
                )
            )
        self.targets.append(
            Target(
                name="Nadir",
                r_LP_P=np.array([0,0.1,0]),
                priority=0,
                Round=0
            )
        )

class Scenario1_2(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        cluster_radius: float = 300000.0,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def regenerate_targets(self) -> None:

        self.targets = []
        satellite = self.satellites[0]
        mu = satellite.sat_args_generator["mu"]  

        num_clusters = 30   
        targets_per_cluster = 1 
        cluster_radius = 300000   


        for task in range(num_clusters):

            f = task * (2 * np.pi / num_clusters)

            cluster_orbit_params = ClassicElements()
            cluster_orbit_params.a = 6771000.0 
            cluster_orbit_params.e = 0         
            cluster_orbit_params.i = 0          
            cluster_orbit_params.Omega = 0     
            cluster_orbit_params.omega = 0    
            cluster_orbit_params.f = f       


            cluster_center_pos, _ = elem2rv(mu, cluster_orbit_params)

            cluster_center_pos = np.array(cluster_center_pos)

            for target_index in range(targets_per_cluster):

                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, cluster_radius)
                x_offset = distance * np.cos(angle)
                y_offset = distance * np.sin(angle)
                z_offset = np.random.uniform(-300000, 300000) 

  
                target_pos = cluster_center_pos + np.array([x_offset, y_offset, z_offset])
                target_pos *= self.radius / np.linalg.norm(target_pos)

                self.targets.append(
                    Target(
                        name=f"cluster_{task}_target_{target_index}",
                        r_LP_P=target_pos,
                        priority=1,
                        #self.priority_distribution(),
                        Round=task  
                    )
                )
            self.targets.append(
                Target(
                    name="Nadir",
                    r_LP_P=np.array([0,0.1,0]),
                    priority=0,
                    Round=0
                )
            )

        os.makedirs("data", exist_ok=True)
        round_num = getattr(self, "current_round", 0) 

        with open("data/Scenario1.csv", mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "x", "y", "z", "priority", "Round"])

            for i, target in enumerate(self.targets):
                r = target.r_LP_P  # [x, y, z]
                name = getattr(target, "name", f"target_{i}")
                priority = getattr(target, "priority", None)
                writer.writerow([name, r[0], r[1], r[2], priority, round_num])

class SparseTarget(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        cluster_radius: float = 300000.0,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius
        self.cluster_radius = cluster_radius
    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def regenerate_targets(self) -> None:

        self.targets = []
        satellite = self.satellites[0]
        mu = satellite.sat_args_generator["mu"]  

        num_clusters = self.n_targets      
        targets_per_cluster = 1  
        cluster_radius = self.cluster_radius   


        for task in range(num_clusters):
 
            f = task * (2 * np.pi / num_clusters)

            cluster_orbit_params = ClassicElements()
            cluster_orbit_params.a = 6771000.0  
            cluster_orbit_params.e = 0          
            cluster_orbit_params.i = 0          
            cluster_orbit_params.Omega = 0      
            cluster_orbit_params.omega = 0     
            cluster_orbit_params.f = f       

            cluster_center_pos, _ = elem2rv(mu, cluster_orbit_params)
            cluster_center_pos = np.array(cluster_center_pos) 

            for target_index in range(targets_per_cluster):

                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, cluster_radius)
                x_offset = distance * np.cos(angle)
                y_offset = distance * np.sin(angle)
                z_offset = np.random.uniform(-cluster_radius, cluster_radius)  

  
                target_pos = cluster_center_pos + np.array([x_offset, y_offset, z_offset])
                target_pos *= self.radius / np.linalg.norm(target_pos)

                self.targets.append(
                    Target(
                        name=f"cluster_{task}_target_{target_index}",
                        r_LP_P=target_pos,
                        priority=1,
                        Round=task
                    )
                )
            self.targets.append(
                Target(
                    name="Nadir",
                    r_LP_P=np.array([0,0.1,0]),
                    priority=0,
                    Round=0
                )
            )
    def reset_during_sim_init(self) -> None:
        """Visualize targets in Vizard on reset."""
        for target in self.targets:
            self.visualize_target(target)

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,  # meters
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1
__doc_title__ = "Target Scenarios"
__all__ = ["Target", "Scenario1_1", "Scenario1_2", "SparseTarget", "Scenario2_2", "DenseTarget","Scenario2_1"]
