import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from experiment_launcher import run_experiment
from Core.cccore import CCCore
from Core.cccore_single import CCCore_single
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger
from datetime import datetime
from imitation_lib.utils import BestAgentSaver
from loco_mujoco import LocoEnv
from Utils.ccutils import get_agent
from mushroom_rl.core.agent import Agent
from pathlib import Path
import sys


env = LocoEnv.make("CCHumanoidTorque.walk")
base_dir = Path("./examples/ccimitation_learning/logs")

file_path = Path("logs") /"loco_mujoco_evaluation_2025-05-15_14-16-39/env_id___CCHumanoidTorque.walk/ \
2/agent_epoch_990_J_40.495957.msh"
agent = Agent.load(str(file_path))
core = CCCore_single(agent,env)


for epoch in range(n_epochs):
    print(f"Starting epoch {epoch}...")
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"{now_str:<20}")
    # train
    core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True, render=False)

    # evaluate
    dataset = core.evaluate(n_episodes=n_eval_episodes)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, gamma=gamma))
    L = np.mean(compute_episodes_length(dataset))
    print(f"R_mean: {R_mean}")
    print(f"J_mean: {J_mean}")
    print(f"L: {L}")
    logger.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
    sw.add_scalar("Eval_R-stochastic", R_mean, epoch)
    sw.add_scalar("Eval_J-stochastic", J_mean, epoch)
    sw.add_scalar("Eval_L-stochastic", L, epoch)
    agent_saver.save(core.agent, R_mean)
agent_saver.save_curr_best_agent()