"""Extended Basilisk SimBaseClass for GeneralSatelliteTasking environments."""
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any
from pathlib import Path
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from time import time
if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.sim.world import WorldModel
from Basilisk.architecture import messaging
from Basilisk.utilities import (
    vizSupport,
)
from Basilisk.utilities import (
    unitTestSupport,
)  # general support file with common unit test functions
import numpy as np
try:
    from Basilisk.simulation import vizInterface

except ImportError:
    pass
logger = logging.getLogger(__name__)
from bsk_rl.utils import vizard
import os
class Simulator(SimulationBaseClass.SimBaseClass):
    """Basilisk simulator for GeneralSatelliteTasking environments."""

    def __init__(
        self,
        satellites: list["Satellite"],
        world_type: type["WorldModel"],
        world_args: dict[str, Any],
        sim_rate: float = 1.0,
        max_step_duration: float = 600.0,
        time_limit: float = float("inf"),
        render = False,
    ) -> None:
        """Basilisk simulator for satellite tasking environments.

        The simulator is reconstructed each time the environment :class:`~bsk_rl.GeneralSatelliteTasking.reset`
        is called, generating a fresh Basilisk simulation.

        Args:
            satellites: Satellites to be simulated
            world_type: Type of world model to be constructed
            world_args: Arguments for world model construction
            sim_rate: [s] Rate for model simulation.
            max_step_duration: [s] Maximum time to propagate sim at a step.
            time_limit: [s] Latest time simulation will propagate to.
        """
        super().__init__()
        self.sim_rate = sim_rate
        self.satellites = satellites
        self.max_step_duration = max_step_duration
        self.time_limit = time_limit
        self.logger = logger

        self.world: WorldModel

        self._set_world(world_type, world_args)

        self.fsw_list = {}
        self.dynamics_list = {}

        for satellite in self.satellites:
            satellite.set_simulator(self)
            self.dynamics_list[satellite.name] = satellite.set_dynamics(self.sim_rate)
            self.fsw_list[satellite.name] = satellite.set_fsw(self.sim_rate)
        if False:
            self.setup_viz()
            self.clear_logs = True
        else:
            self.clear_logs = True
    def finish_init(self) -> None:
        """Finish simulator initialization."""
        self.set_vizard_epoch()
        self.InitializeSimulation()
        self.ConfigureStopTime(0)
        self.ExecuteSimulation()

    @property
    def sim_time_ns(self) -> int:
        """Simulation time in ns, tied to SimBase integrator."""
        return self.TotalSim.CurrentNanos

    @property
    def sim_time(self) -> float:
        """Simulation time in seconds, tied to SimBase integrator."""
        return self.sim_time_ns * mc.NANO2SEC
    @vizard.visualize
    def setup_vizard(self, vizard_rate=None, vizSupport=None, **vizard_settings):
        """Setup Vizard for visualization."""
        if hasattr(self, "vizInstance") and self.vizInstance is not None:
            del self.vizInstance
            print("dell")
        vizard.VIZINSTANCE = None  # Avoid global pollution
        save_path = Path(vizard.VIZARD_PATH)
        if not save_path.exists():
            os.makedirs(save_path, exist_ok=True)

        viz_proc_name = "VizProcess"
        viz_proc = self.CreateNewProcess(viz_proc_name, priority=400)

        # Define process name, task name and task time-step
        viz_task_name = "viz_task_name"
        if vizard_rate is None:
            vizard_rate = self.sim_rate
        viz_proc.addTask(self.CreateNewTask(viz_task_name, mc.sec2nano(vizard_rate)))

        customizers = ["spriteList", "genericSensorList"]
        list_data = {}
        for customizer in customizers:
            list_data[customizer] = [
                sat.vizard_data.get(customizer, None) for sat in self.satellites
            ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.vizInstance = vizSupport.enableUnityVisualization(
            self,
            viz_task_name,
            scList=[sat.dynamics.scObject for sat in self.satellites],
            **list_data,
            liveStream=False,
            saveFile = f"episode_{timestamp}"
        )
        for key, value in vizard_settings.items():
            setattr(self.vizInstance.settings, key, value)
        vizard.VIZINSTANCE = self.vizInstance

    @vizard.visualize
    def set_vizard_epoch(self, vizInstance=None):
        """Set the Vizard epoch."""
        vizInstance.epochInMsg.subscribeTo(self.world.gravFactory.epochMsg)

    def _set_world(
        self, world_type: type["WorldModel"], world_args: dict[str, Any]
    ) -> None:
        """Construct the simulator world model.

        Args:
            world_type: Type of world model to be constructed.
            world_args: Arguments for world model construction, passed to the world
                from the environment.
        """
        self.world = world_type(self, self.sim_rate, **world_args)

    def run(self) -> None:
        """Propagate the simulator.

        Propagates for a duration up to the ``max_step_duration``, stopping if the
        environment time limit is reached or an event is triggered.
        """
        if "max_step_duration" in self.eventMap:
            self.delete_event("max_step_duration")

        self.createNewEvent(
            "max_step_duration",
            mc.sec2nano(self.sim_rate),
            True,
            [
                f"self.TotalSim.CurrentNanos * {mc.NANO2SEC} >= {self.sim_time + self.max_step_duration}"
            ],
            ["self.logger.info('Max step duration reached')"],
            terminal=True,
        )
        self.ConfigureStopTime(mc.sec2nano(min(self.time_limit, 2**31)))
        self.ExecuteSimulation()

    def delete_event(self, event_name) -> None:
        """Remove an event from the event map.

        Makes event checking faster. Due to a performance issue in Basilisk, it is
        necessary to remove created for tasks that are no longer needed (even if it is
        inactive), or else significant time is spent processing the event at each step.
        """
        event = self.eventMap[event_name]
        self.eventList.remove(event)
        del self.eventMap[event_name]

    def __del__(self):
        """Log when simulator is deleted."""
        logger.debug("Basilisk simulator deleted")
    def setup_viz(self):
        """
        Initializes a vizSupport instance and logs all RW/thruster/spacecraft
        state messages.
        """
        scObjects = []
        for sat_dyn in self.dynamics_list.values():
            scObjects.append(sat_dyn.scObject)
        hdLists = [None]*len(self.dynamics_list)  # Initialize an empty list for all panels

        count = 0
        for sat_name, sat_dyn in self.dynamics_list.items():
            # Create battery panel
            batteryPanel = vizInterface.GenericStorage()
            batteryPanel.label = f"Battery_{sat_name}"
            batteryPanel.units = "Ws"
            batteryPanel.thresholds = vizInterface.IntVector([50])
            batteryPanel.color = vizInterface.IntVector(
                vizSupport.toRGBA255("blue") + vizSupport.toRGBA255("red")
            )
            batteryInMsg = messaging.PowerStorageStatusMsgReader()
            batteryInMsg.subscribeTo(self.dynamics_list[sat_name].powerMonitor.batPowerOutMsg)
            batteryPanel.batteryStateInMsg = batteryInMsg
            batteryPanel.this.disown()
            # Create data monitor panel
            dataMonitor = vizInterface.GenericStorage()
            dataMonitor.label = f"Data Monitor_{sat_name}"
            dataMonitor.units = "MB"
            dataMonitor.thresholds = vizInterface.IntVector([50])
            dataMonitor.color = vizInterface.IntVector(
                vizSupport.toRGBA255("red") + vizSupport.toRGBA255("blue")
            )
            dataInMsg = messaging.DataStorageStatusMsgReader()
            dataInMsg.subscribeTo(self.dynamics_list[sat_name].storageUnit.storageUnitDataOutMsg)
            dataMonitor.dataStorageStateInMsg = dataInMsg
            dataMonitor.this.disown()
            # Create thruster power panel
           # hdDevicePanel = vizInterface.GenericStorage()
            #hdDevicePanel.label = f"Main Disk_{sat_dyn.thrusterPowerSink.ModelTag}"
           # hdDevicePanel.units = "W"
            #hdDevicePanel.thresholds = vizInterface.IntVector([50])
            #hdDevicePanel.color = vizInterface.IntVector(
             #   vizSupport.toRGBA255("blue") + vizSupport.toRGBA255("red")
            #)
            #hdInMsg = messaging.PowerNodeUsageMsgReader()
           # hdInMsg.subscribeTo(self.dynamics_list[sat_name].thrusterPowerSink.nodePowerOutMsg)
            # hdDevicePanel.powerUsageMsg = hdInMsg

            # Append the created panels to the list
            hdLists[count]=[batteryPanel, dataMonitor]
            count = count + 1
        # Now hdLists contains a list of panels for each satellite





        self.vizInterface = vizSupport.enableUnityVisualization(
            self,
            sat_dyn.task_name,
            scObjects,
            genericStorageList=hdLists,
            liveStream=False,
            saveFile='test',
        )
        satellite = self.satellites[0]
        for target in satellite.known_targets:
            r_LP_P_Init = [[item] for item in target.r_LP_P]
            sat_dyn.imagingTarget.r_LP_P_Init = r_LP_P_Init
            vizSupport.addLocation(
                self.vizInterface,
                stationName=str("Target"),
                parentBodyName="earth",
                r_GP_P=unitTestSupport.EigenVector3d2list(sat_dyn.imagingTarget.r_LP_P_Init),
                fieldOfView=sat_dyn.imagingTarget.minimumElevation,
                color="red",
                range=sat_dyn.imagingTarget.maximumRange,  # meters
            )
        vizSupport.createPointLine(self.vizInterface, toBodyName="sun", lineColor="yellow")
        vizSupport.setInstrumentGuiSetting(self.vizInterface, showGenericStoragePanel=True)
        self.vizInterface.settings.spacecraftSizeMultiplier=0.0001
        self.vizInterface.settings.showLocationCones=-1
        
__all__ = []
