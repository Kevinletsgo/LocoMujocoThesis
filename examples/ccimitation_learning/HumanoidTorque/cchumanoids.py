# from loco_mujoco.environments.humanoids.humanoids import HumanoidTorque
# import numpy as np
# from mushroom_rl.utils.spaces import Box
# from loco_mujoco.utils import check_validity_task_mode_dataset
# from loco_mujoco.environments.humanoids.base_humanoid import CCBaseHumanoid
#
# class CCHumanoidTorque(HumanoidTorque):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     @staticmethod
#     def generate(task="walk", dataset_type="real", **kwargs):
#
#         check_validity_task_mode_dataset(HumanoidTorque.__name__, task, None, dataset_type,
#                                          *HumanoidTorque.valid_task_confs.get_all())
#
#         if dataset_type == "real":
#             if task == "walk":
#                 path = "datasets/humanoids/real/02-constspeed_reduced_humanoid.npz"
#             elif task == "run":
#                 path = "datasets/humanoids/real/05-run_reduced_humanoid.npz"
#         elif dataset_type == "perfect":
#             if "use_foot_forces" in kwargs.keys():
#                 assert kwargs["use_foot_forces"] is False
#             if "disable_arms" in kwargs.keys():
#                 assert kwargs["disable_arms"] is True
#             if "use_box_feet" in kwargs.keys():
#                 assert kwargs["use_box_feet"] is True
#
#             if task == "walk":
#                 path = "datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz"
#             elif task == "run":
#                 path = "datasets/humanoids/perfect/humanoid_torque_run/perfect_expert_dataset_det.npz"
#         #
#         # expert_data = np.load(path)
#         # print(expert_data)
#         return BaseHumanoid.generate(CCHumanoidTorque, path, task, dataset_type, **kwargs)