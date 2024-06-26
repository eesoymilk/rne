import os

import cv2
import torch
import numpy as np
from tqdm import tqdm

import wrapper
from model import PolicyNet


def main():
    # Parameters that are fixed
    # ----------------------------
    n_iter = 100
    s_dim = 14
    a_dim = 1
    save_dir = './save'
    device = 'cpu'

    # Create environment & model
    # ----------------------------
    env = wrapper.PathTrackingEnv()
    policy_net = PolicyNet(s_dim, a_dim).to(device)

    # Load model
    # ----------------------------
    if os.path.exists(os.path.join(save_dir, "model.pt")):
        print("Loading the model ... ", end="")
        checkpoint = torch.load(os.path.join(save_dir, "model.pt"))
        policy_net.load_state_dict(checkpoint["PolicyNet"])
        print("Done.")
    else:
        print("Error: No model saved")

    # Start playing
    # ----------------------------
    policy_net.eval()
    mean_total_reward = 0

    for it in tqdm(range(n_iter), desc="Evaluating"):
        ob, _ = env.reset()
        total_reward = 0
        length = 0

        while True:
            # Step
            state_tensor = torch.tensor(
                np.expand_dims(ob, axis=0), dtype=torch.float32, device=device
            )
            action = (
                policy_net.action_step(state_tensor, deterministic=True)
                .cpu()
                .detach()
                .numpy()
            )
            ob, reward, done, info = env.step(action[0])
            total_reward += reward
            length += 1
            if done:
                mean_total_reward += total_reward
                break

        # print(f"{total_reward = :.6f}, {length = :d}", flush=True)

    mean_total_reward /= n_iter
    print(f"Evaluation Score: {mean_total_reward:.4f}")


if __name__ == '__main__':
    main()
