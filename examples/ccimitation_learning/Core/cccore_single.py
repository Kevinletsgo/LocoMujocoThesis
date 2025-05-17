import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm
from collections import defaultdict
from mushroom_rl.utils.record import VideoRecorder
from mushroom_rl.core.core import Core

class CCCore_single(Core):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actionTorque_names = ['mot_lumbar_extension', 'mot_lumbar_bending', 'mot_lumbar_rotation',
                              'mot_hip_flexion_r', 'mot_hip_adduction_r', 'mot_hip_rotation_r',
                              'mot_knee_angle_r', 'mot_ankle_angle_r',
                              'mot_hip_flexion_l', 'mot_hip_adduction_l', 'mot_hip_rotation_l',
                              'mot_knee_angle_l', 'mot_ankle_angle_l']

        self.actionParam_names = [
            "k_lumbar_extension", "q_ref_lumbar_extension", "d_lumbar_extension",

            "k_hip_flexion_stance", "q_ref_hip_flexion_stance", "d_hip_flexion_stance", "c_hip_flexion",
            "q_0_hip_flexion",
            "k_hip_flexion_swing", "q_ref_hip_flexion_swing", "d_hip_flexion_swing",

            "k_knee_angle_stance", "q_ref_knee_angle_stance", "d_knee_angle_stance", "c_knee_angle",
            "q_0_knee_angle",
            "k_knee_angle_swing", "d_knee_angle_swing", "q_ref_kne1_swing", "q_ref_kne2_swing",
            "q_hip_th",

            "k_ankle_angle_stance", "q_ref_ankle_angle_stance", "d_ankle_angle_stance", "c_ankle_angle",
            "q_0_ankle_angle",
            "k_ankle_angle_swing", "q_ref_ankle_angle_swing", "d_ankle_angle_swing",

            "mot_lumbar_bending", "mot_lumbar_rotation",
            "mot_hip_adduction_r", "mot_hip_rotation_r",
            "mot_hip_adduction_l", "mot_hip_rotation_l"
        ]

        self.state_names = [
            'q_pelvis_ty',
            'q_pelvis_tilt',
            'q_pelvis_list',
            'q_pelvis_rotation',
            'q_hip_flexion_r',
            'q_hip_adduction_r',
            'q_hip_rotation_r',
            'q_knee_angle_r',
            'q_ankle_angle_r',
            'q_hip_flexion_l',
            'q_hip_adduction_l',
            'q_hip_rotation_l',
            'q_knee_angle_l',
            'q_ankle_angle_l',
            'q_lumbar_extension',
            'q_lumbar_bending',
            'q_lumbar_rotation',
            'dq_pelvis_tx',
            'dq_pelvis_tz',
            'dq_pelvis_ty',
            'dq_pelvis_tilt',
            'dq_pelvis_list',
            'dq_pelvis_rotation',
            'dq_hip_flexion_r',
            'dq_hip_adduction_r',
            'dq_hip_rotation_r',
            'dq_knee_angle_r',
            'dq_ankle_angle_r',
            'dq_hip_flexion_l',
            'dq_hip_adduction_l',
            'dq_hip_rotation_l',
            'dq_knee_angle_l',
            'dq_ankle_angle_l',
            'dq_lumbar_extension',
            'dq_lumbar_bending',
            'dq_lumbar_rotation'
        ]
        self.action_index = {name: idx for idx, name in enumerate(self.actionParam_names)}
        self.state_index = {name: idx for idx, name in enumerate(self.state_names)}
    def _step(self, render, record):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the absorbing flag of the reached state and the last step flag.

        """
        # TODO here should be action parameters
        # action = self.agent.draw_action(self._state)
        self.GRF = self.mdp._get_ground_forces()

        normalized_actionParam = self.agent.draw_action(self._state)
        unnormalized_actionParam = self._preprocess_actionParam(normalized_actionParam)

        # unnormalized action
        action = self.getAction(unnormalized_actionParam, self._state)

        next_state, reward, absorbing, step_info = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record)

            if record:
                self._record(frame)

        last = not (
                self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return (state, normalized_actionParam, reward, next_state, absorbing, last), step_info

    def _preprocess_actionParam(self, actionParam):
        """
        This function preprocesses all actions. All actions in this environment expected to be between -1 and 1.
        Hence, we need to unnormalize the action to send to correct action to the simulation.
        Note: If the action is not in [-1, 1], the unnormalized version will be clipped in Mujoco.

        Args:
            action (np.array): Action to be send to the environment;

        Returns:
            Unnormalized action (np.array) that is send to the environment;

        """

        unnormalized_actionParam = ((actionParam.copy() * self.mdp.norm_act_delta_param) + self.mdp.norm_act_mean_param)
        return unnormalized_actionParam

    def getAction(self, actionParam, state):
        # [foot_r,foot_l]
        # GRF_x_r = self.GRF[2]
        # GRF_x_l = self.GRF[5]
        #
        # GRF_z_r = self.GRF[0]
        # GRF_z_l = self.GRF[3]
        GRF_x_r = state[38] * 1000
        GRF_x_l = state[41] * 1000

        GRF_z_r = state[36] * 1000
        GRF_z_l = state[39] * 1000
        # GRF_x_r = 0
        # GRF_x_l = 0
        #
        # GRF_z_l = 0
        # GRF_z_r = 0
        mot_lumbar_extension = self.calculate_lumbar_extension(actionParam, state)
        mot_hip_flexion_r = self.calculate_hip_flexion(actionParam, state, GRF_z_r, is_left=0)
        mot_hip_flexion_l = self.calculate_hip_flexion(actionParam, state, GRF_z_l, is_left=1)
        mot_knee_angle_r = self.calculate_knee_angle(actionParam, state, GRF_z_r, is_left=0)
        mot_knee_angle_l = self.calculate_knee_angle(actionParam, state, GRF_z_l, is_left=1)
        mot_ankle_angle_r = self.calculate_ankle_angle(actionParam, state, GRF_z_r, GRF_x_r, is_left=0)
        mot_ankle_angle_l = self.calculate_ankle_angle(actionParam, state, GRF_z_l, GRF_x_l, is_left=1)

        # mot_lumbar_bending = 0
        # mot_lumbar_rotation = 0
        # mot_hip_adduction_r = 0
        # mot_hip_adduction_l = 0
        # mot_hip_rotation_r = 0
        # mot_hip_rotation_l = 0
        mot_lumbar_bending = actionParam[self.action_index['mot_lumbar_bending']]
        mot_lumbar_rotation = actionParam[self.action_index['mot_lumbar_rotation']]
        mot_hip_adduction_r = actionParam[self.action_index['mot_hip_adduction_r']]
        mot_hip_adduction_l = actionParam[self.action_index['mot_hip_adduction_l']]
        mot_hip_rotation_r = actionParam[self.action_index['mot_hip_rotation_r']]
        mot_hip_rotation_l = actionParam[self.action_index['mot_hip_rotation_l']]
        self.actionTorque_names = ['mot_lumbar_extension', 'mot_lumbar_bending', 'mot_lumbar_rotation',
                              'mot_hip_flexion_r', 'mot_hip_adduction_r', 'mot_hip_rotation_r',
                              'mot_knee_angle_r', 'mot_ankle_angle_r',
                              'mot_hip_flexion_l', 'mot_hip_adduction_l', 'mot_hip_rotation_l',
                              'mot_knee_angle_l', 'mot_ankle_angle_l']
        joint_torques = np.array([
            mot_lumbar_extension, mot_lumbar_bending, mot_lumbar_rotation,
            mot_hip_flexion_r, mot_hip_adduction_r, mot_hip_rotation_r,
            mot_knee_angle_r, mot_ankle_angle_r,
            mot_hip_flexion_l, mot_hip_adduction_l, mot_hip_rotation_l,
            mot_knee_angle_l, mot_ankle_angle_l])

        return joint_torques
    def calculate_lumbar_extension(self, actionParam, state):
        return self.calculate_lumbar_torque(actionParam, state)

    def calculate_hip_flexion(self, actionParam, state, grfz, is_left):
        return self.calculate_hip_torque("hip_flexion", actionParam, state, grfz, is_left)

    def calculate_knee_angle(self, actionParam, state, grfz, is_left):
        return self.calculate_knee_torque("knee_angle", actionParam, state, grfz, is_left)

    def calculate_ankle_angle(self, actionParam, state, grfz, grfx, is_left):
        return self.calculate_ankle_torque("ankle_angle", actionParam, state, grfz, grfx, is_left)

    def leg_in_stance(self, grf, threshold = 20):
        grf = 0 if grf <= threshold else grf
        return grf

    def get_joint_params(self, joint_name, actionParam, state, is_stance, is_left):
        # stance phase
        k, q_ref, d, c, q_0 = self._get_joint_action_parameters(joint_name, actionParam, is_stance)
        if is_left:
            q, dq = self._get_joint_states_l(joint_name, state)
        else:
            q, dq = self._get_joint_states_r(joint_name, state)
        return k, q_ref, d, c, q_0, q, dq

    def _get_joint_action_parameters(self, joint_name, actionParam, is_stance):
        """提取所有关节参数与状态"""
        if is_stance:
            # stance phase
            k = actionParam[self.action_index[f'k_{joint_name}_stance']]
            q_ref = actionParam[self.action_index[f'q_ref_{joint_name}_stance']]
            d = actionParam[self.action_index[f'd_{joint_name}_stance']]
            c = actionParam[self.action_index[f'c_{joint_name}']]
            q_0 = actionParam[self.action_index[f'q_0_{joint_name}']]
        else:
            # swing phase
            k = actionParam[self.action_index[f'k_{joint_name}_swing']]
            q_ref = actionParam[self.action_index[f'q_ref_{joint_name}_swing']]
            d = actionParam[self.action_index[f'd_{joint_name}_swing']]
            c = 0
            q_0 = 0
        return k, q_ref, d, c, q_0


    def _get_joint_states_r(self, joint_name, state):
        """提取所有关节参数与状态"""
        q = state[self.state_index[f'q_{joint_name}_r']]
        dq = state[self.state_index[f'dq_{joint_name}_r']]
        return q, dq

    def _get_joint_states_l(self, joint_name, state):
        """提取所有关节参数与状态"""
        q = state[self.state_index[f'q_{joint_name}_l']]
        dq = state[self.state_index[f'dq_{joint_name}_l']]
        return q, dq

    def _get_swing_knee_q_ref(self, actionParam, is_left):
        if is_left:
            q_hip = actionParam[self.state_index['q_hip_flexion_l']]
        else:
            q_hip = actionParam[self.state_index['q_hip_flexion_r']]
        q_hip_th = actionParam[self.action_index['q_hip_th']]
        key = 'q_ref_kne1_swing' if q_hip < q_hip_th else 'q_ref_kne2_swing'
        return actionParam[self.action_index[key]]

    def calculate_hip_torque(self, joint_name, actionParam, state, grfz, is_left):
        if self.leg_in_stance(grfz):
            is_stance = 1
        else:
            is_stance = 0
        k, q_ref, d, c, q_0, q, dq = self.get_joint_params(joint_name, actionParam, state, is_stance, is_left)

        if is_stance:
            torque = k * (q_ref - q) + d * dq + c * grfz * (q_0 - q)
        else:
            torque = k * (q_ref - q) + d * dq
        return torque

    # to optimize
    def calculate_knee_torque(self, joint_name, actionParam, state, grfz, is_left):
        if self.leg_in_stance(grfz):
            is_stance = 1
            k, q_ref, d, c, q_0, q, dq = self.get_joint_params(joint_name, actionParam, state, is_stance, is_left)
            torque = k * (q_ref - q) + d * dq + c * grfz * (q_0 - q)
        else:
            k = actionParam[self.action_index[f'k_{joint_name}_swing']]
            d = actionParam[self.action_index[f'd_{joint_name}_swing']]
            q_ref = self._get_swing_knee_q_ref(actionParam, is_left)
            if is_left:
                q = state[self.state_index[f'q_{joint_name}_l']]
                dq = state[self.state_index[f'dq_{joint_name}_l']]
            else:
                q = state[self.state_index[f'q_{joint_name}_r']]
                dq = state[self.state_index[f'dq_{joint_name}_r']]
            torque = k * (q_ref - q) + d * dq
        return torque


    def calculate_ankle_torque(self, joint_name, actionParam, state, grfx, grfz, is_left):
        if self.leg_in_stance(grfz):
            is_stance = 1
        else:
            is_stance = 0
        k, q_ref, d, c, q_0, q, dq = self.get_joint_params(joint_name, actionParam, state, is_stance, is_left)
        if self.leg_in_stance(grfz):
            # stance phase
            torque = k * (q_ref - q) + d * dq + c * max(grfx, 0) * (q_0 - q)
        else:
            # swing phase
            torque = k * (q_ref - q) + d * dq
        return torque

    def calculate_lumbar_torque(self, actionParam, state):
        k = actionParam[self.action_index[f'k_lumbar_extension']]
        q_ref = actionParam[self.action_index[f'q_ref_lumbar_extension']]
        d = actionParam[self.action_index[f'd_lumbar_extension']]
        q = state[self.state_index[f'q_lumbar_extension']] #state[14]
        dq = state[self.state_index[f'dq_lumbar_extension']] #state[33]
        return k * (q_ref - q) + d * dq
