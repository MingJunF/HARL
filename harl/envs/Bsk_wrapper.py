import os
import numpy as np
import pandas as pd
from functools import partial
from gymnasium import spaces
from Basilisk.utilities import orbitalMotion, macros
from Basilisk.utilities.orbitalMotion import ClassicElements
from Basilisk.architecture import bskLogging

from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.data.unique_image_data import UniqueImageReward
from bsk_rl.data.revisitImageData import RevisitImageReward
from bsk_rl.gym import ConstellationTasking
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
class MultiAgentEnv(object):
    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list"""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the shape of the observation"""
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError


    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

    def get_stats(self):
        return {}

class CustomSatComposed(sats.ImagingSatellite):
    observation_spec = [
        obs.Time(),
        obs.SatProperties(
            dict(prop="omega_BP_P", norm=0.03),
            dict(prop="c_hat_P"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
            dict(prop="FOV"),         
        ),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm=6300),
            dict(prop="opportunity_close", norm=6300),
            dict(prop="priority"),
            dict(prop="r_LP_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="target_angle", norm=np.pi),
            n_ahead_observe=7,
        ),
        obs.NearbySatellitesAttitude()
    ]

    action_spec = [
        act.Image(n_ahead_image=27),
    ]

    class CustomDynModel(dyn.FullFeaturedDynModel):
        @property
        def solar_angle_norm(self) -> float:
            sun_vec_N = self.world.gravFactory.spiceObject.planetStateOutMsgs[
                self.world.sun_index
            ].read().PositionVector
            sun_vec_N_hat = sun_vec_N / np.linalg.norm(sun_vec_N)
            solar_panel_vec_B = np.array([0, 0, -1])
            mat = self.BN.T
            solar_panel_vec_N = mat @ solar_panel_vec_B
            error_angle = np.arccos(np.clip(np.dot(solar_panel_vec_N, sun_vec_N_hat), -1.0, 1.0))
            return error_angle / np.pi

    dyn_type = CustomDynModel
    fsw_type = fsw.UniqueImagerFSWModel

def create_env(map_name, Target_type="SparseTarget", Num_targets=60,Target_density=400000,  Sat_orb_param="2SatCluster.xlsx",render=False):
    file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../Satellites", Sat_orb_param))
    df = pd.read_excel(file_path)

    satellites = []
    for _, row in df.iterrows():
        name = row["name"].replace(" ", "_")
        oe = ClassicElements()
        oe.a = row["k"] * 1000
        oe.e = row["e"]
        oe.i = row["i"] * macros.D2R
        oe.Omega = row["\u03a9"] * macros.D2R
        oe.omega = row["\u03c9"] * macros.D2R
        oe.f = row["M"] * macros.D2R

        sat_args = CustomSatComposed.default_sat_args(
            oe=oe,
            imageAttErrorRequirement=1,
            imageRateErrorRequirement=1,
            batteryStorageCapacity=80.0 * 3600 * 40,
            storedCharge_Init=np.random.uniform(1, 1) * 80.0 * 3600 * 40,
            u_max=0.2,
            K1=0.5,
            dataStorageCapacity=100,
            nHat_B=np.array([1, 0, 0]),
            imageTargetMinimumElevation=np.radians(20),
            rwBasePower=20,
            maxWheelSpeed=1500,
            storageInit=0,
            wheelSpeeds=np.random.uniform(-1, 1, 3),
        )
        satellites.append(CustomSatComposed(name, sat_args))

    scenario_module = __import__("bsk_rl.scene.targets", fromlist=[Target_type])
    scenario_cls = getattr(scenario_module, Target_type)
    scenario = scenario_cls(n_targets=Num_targets,cluster_radius=Target_density)
    rewarder = UniqueImageReward()

    env = ConstellationTasking(
        satellites=satellites,
        scenario=scenario,
        rewarder=rewarder,
        terminate_on_time_limit=True,
        world_type=world.GroundStationWorldModel,
        world_args=world.GroundStationWorldModel.default_world_args(),
        sim_rate=1,
        time_limit=6300,
        log_level="WARN",
        failure_penalty=-10,
        render=render

    )
    return env, Num_targets

class BSKWrapper(MultiAgentEnv):
    def __init__(self, env_args):
        self.map_name = env_args.get("map_name", "Scenario1")
        self.Target_type = env_args.get("Target_type", "SparseTarget")
        self.Num_targets = env_args.get("Num_targets", 60)
        self.Target_density = env_args.get("Target_density", 400000)
        self.Sat_orb_param = env_args.get("Sat_orb_param", "2SatCluster.xlsx")
        self.render = env_args.get("render", False)

        # 🔒 安全检查
        assert isinstance(self.Sat_orb_param, str), f"Sat_orb_param must be a string, got {type(self.Sat_orb_param)}"

        # 🌍 创建真实环境
        self.env, self.total_targets = create_env(
            map_name=self.map_name,
            Sat_orb_param=self.Sat_orb_param,
            Target_type=self.Target_type,
            Target_density=self.Target_density,
            render=self.render,
        )
        self.episode_limit = self.env.episode_limit
        self.n_agents = len(self.env.satellites)
        self.env.reset()


    def _get_rewarder_cls(self, rewarder_name):
        from bsk_rl.data import unique_image_data, revisitImageData
        if hasattr(unique_image_data, rewarder_name):
            return getattr(unique_image_data, rewarder_name)
        elif hasattr(revisitImageData, rewarder_name):
            return getattr(revisitImageData, rewarder_name)
        else:
            raise ValueError(f"Rewarder {rewarder_name} not found in bsk_rl.data.unique_image_data or bsk_rl.data.revisitImageData")


    def step(self, actions):
        actions_dict = {self.env.agents[i]: int(actions[i]) for i in range(len(actions))}
        observation, reward_dict, terminated, truncated, info, completeness = self.env.step(actions_dict)
        total_reward = sum(reward_dict.values())
        info_dict = {}
        if terminated:
            info_dict["Task_completion_rate"] = len(completeness) / self.total_targets
        terminated = any(terminated.values()) or any(truncated.values())
        return observation, total_reward, terminated, truncated, info_dict

    def get_obs(self):
        return [self.env._get_obs()[agent_id] for agent_id in self.env.possible_agents]

    def get_obs_agent(self, agent_id):
        return self.env._get_obs()[self.env.possible_agents[agent_id]]

    def get_obs_size(self):
        return self.env.observation_space(self.env.possible_agents[0]).shape[0]

    def get_state(self):
        return np.concatenate([self.get_obs_agent(i) for i in range(self.n_agents)])

    def get_state_size(self):
        return self.get_state().shape[0]

    def get_avail_agent_actions(self, agent_id):
        agent_name = self.env.possible_agents[agent_id]
        agent2sat = dict(zip(self.env.possible_agents, self.env.satellites))
        satellite = agent2sat[agent_name]
        n = self.env.action_space(agent_name).n
        if getattr(satellite, "using_dummy_padding", False):
            return [1] + [1] * (n - 1)
        else:         
            return [1] + [1] * (n - 1)


    def get_avail_actions(self):
        return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]

    def get_total_actions(self):
        return len(self.get_avail_agent_actions(0))

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)


    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_print_info(self):
        return None
    @property
    def observation_space(self):
        return self.env.observation_space  
    @property
    def action_space(self):
        return self.env.action_space
    @property
    def share_observation_space(self):
        return self.env.action_space