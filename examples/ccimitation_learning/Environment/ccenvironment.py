# from mushroom_rl.core import MDPInfo
#
#
# class CCMDPInfo(MDPInfo):
#     def __init__(self, observation_space, action_space, actionParam_space, gamma, horizon, dt=1e-1):
#         super().__init__(observation_space, action_space, gamma, horizon, dt)
#         self.actionParam_space = actionParam_space
#         # 先清空（假设 _save_attr 是个 dict 或类似结构）
#         self._save_attr.clear()
#
#         # 只添加你想保存的字段，不包括 action_space
#         self._add_save_attr(
#             observation_space='mushroom',
#             actionParam_space='mushroom',
#             gamma='primitive',
#             horizon='primitive',
#             dt='primitive'
#         )
#     @property
#     def size(self):
#         """
#         Returns:
#             The sum of the number of discrete states and discrete actions. Only works for discrete spaces.
#
#         """
#         return self.observation_space.size + self.actionParam_space.size
#
#     @property
#     def shape(self):
#         """
#         Returns:
#             The concatenation of the shape tuple of the state and action spaces.
#
#         """
#         return self.observation_space.shape + self.actionParam_space_space.shape