import sys
from pathlib import Path

# اضافه کردن روت پروژه به مسیر پایتون برای جلوگیری از ارور ایمپورت
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from agents.sac.sac_agent import SACAgent
from rl.sac.replay_buffer import SACReplayBuffer
from config import sac_config as cfg

def test_local_sac():
    print("[INFO] Starting Local SAC Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ساخت ایجنت و بافر
    agent = SACAgent(device=device)
    buffer = SACReplayBuffer(capacity=1000, device=device)
    
    # تولید دیتای دامی بر اساس کانفیگ واقعی پروژه
    print("[INFO] Populating Replay Buffer with Dummy Data...")
    for _ in range(300):
        dummy_grid = np.random.randn(5, 25, 11).astype(np.float32)
        dummy_scalar = np.random.randn(8).astype(np.float32) # تغییر به ۸ اسکالر واقعی
        dummy_action = np.random.uniform(-1, 1, size=(3,)).astype(np.float32)
        dummy_reward = np.random.randn()
        dummy_next_grid = np.random.randn(5, 25, 11).astype(np.float32)
        dummy_next_scalar = np.random.randn(8).astype(np.float32) # تغییر به ۸ اسکالر واقعی
        dummy_done = 0.0
        
        buffer.add(dummy_grid, dummy_scalar, dummy_action, dummy_reward, 
                   dummy_next_grid, dummy_next_scalar, dummy_done)
    
    print(f"[INFO] Testing SAC Updates (Warmup steps: {cfg.WARMUP_STEPS})")
    for step in range(1, 15):
        losses = agent.update(buffer)
        print(f"Step {step} | Critic Loss: {losses['critic_loss']:.4f} | Actor Loss: {losses['actor_loss']:.4f}")

if __name__ == "__main__":
    test_local_sac()