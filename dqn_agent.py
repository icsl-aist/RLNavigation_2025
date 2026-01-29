#!/usr/bin/env python3
#################################################################################
# PPO (Continuous Action) replacement for original DQN agent — SB3-style refactor
# - Behavior preserved 100% (ROS/Gazebo I/F, math, schedules, logging)
# - SB3-like naming/structure: learn(), train(), RolloutBuffer, clip_range, n_steps
#################################################################################

import os
import sys
import time
import json
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn as DqnSrv

import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import glob
import re


dt = datetime.datetime.today()

# ============================================================
# ユーティリティ関数群（挙動は元コードと同一）
# ============================================================
def to_tensor(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)

def compute_gae(rewards, values, is_last_step_terminal, gamma=0.99, lam=0.95):
    T = len(rewards) # rewards の長さは T
    # values の長さは T+1 (V(s_0) ~ V(s_T))
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)): 
        is_terminal = (t == T - 1) and is_last_step_terminal
        next_value = 0.0 if is_terminal else values[t + 1] 
        
        delta = rewards[t] + gamma * next_value - values[t] # values[t] は V(s_t)
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        
    returns = (advantages + np.asarray(values[:-1], dtype=np.float32)).tolist() # ★ 修正: values[:-1] を使用
    return returns, advantages.tolist()

def normalize_tensor(x: torch.Tensor, eps: float = 1e-8):
    if x.numel() == 0:
        return x
    mean = x.mean()
    std = x.std(unbiased=False)
    if float(std) < eps:
        return x - mean
    return (x - mean) / (std + eps)


# ============================================================
# ニューラルネットワーク定義（元コードそのまま）
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim=28, action_dim=2):  # 2次元: [linear, angular]
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(128, action_dim)
        # バッチ独立 log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.net(x)
        mu = torch.tanh(self.mu_layer(x))                 # 出力を安全に[-1,1]
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)    # 0や∞を回避
        return mu, std

class Critic(nn.Module):
    def __init__(self, state_dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


# ============================================================
# SB3風 RolloutBuffer（挙動は元の配列運用と完全一致）
# - 1エピソード分を受け取り、内部配列に extend
# - 先頭 n_steps を「学習に使用する部分」として抽出できるAPIを提供
# ============================================================
class RolloutBuffer:
    def __init__(self):
        self.states   = []
        self.actions  = []
        self.logps    = []
        self.returns  = []
        self.advantages = []
        self.values   = []
        self.collected_steps = 0

    def add_episode(self, ep_states, ep_actions, ep_logps, ep_values, ep_returns, ep_advs):
        # 1エピソード分をそのまま後ろに積む（元コードの extend と同じ）
        self.states.extend(ep_states)
        self.actions.extend(ep_actions)
        self.logps.extend(ep_logps)
        self.values.extend(ep_values)
        self.returns.extend(ep_returns)
        self.advantages.extend(ep_advs)
        self.collected_steps += len(ep_rewards := ep_returns)  # returns と rewards 長さは同じ

    def size(self):
        return self.collected_steps

    def take_head(self, n_steps):
        """
        先頭 n_steps 分を '使用対象' として切り出し、残りは末尾に温存（元コードと同じ）
        """
        used = {
            "states":      self.states[:n_steps],
            "actions":     self.actions[:n_steps],
            "logps":       self.logps[:n_steps],
            "returns":     self.returns[:n_steps],
            "advantages":  self.advantages[:n_steps],
            "values":      self.values[:n_steps],
        }
        extra_steps = self.collected_steps - n_steps
        if extra_steps > 0:
            # 余り（末尾の extra_steps 件）だけ残す：元コードそのままの挙動
            self.states     = self.states[-extra_steps:]
            self.actions    = self.actions[-extra_steps:]
            self.logps      = self.logps  [-extra_steps:]
            self.returns    = self.returns[-extra_steps:]
            self.advantages = self.advantages[-extra_steps:]
            self.values     = self.values [-extra_steps:]
        else:
            # ちょうど n_steps のときは空にする
            self.states.clear(); self.actions.clear(); self.logps.clear()
            self.returns.clear(); self.advantages.clear(); self.values.clear()
        self.collected_steps = max(0, extra_steps)
        return used, extra_steps

    def iter_minibatches(self, used_dict, batch_size):
        num_samples = len(used_dict["states"])
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield (
                [used_dict["states"][i]     for i in idx],
                [used_dict["actions"][i]    for i in idx],
                [used_dict["logps"][i]      for i in idx],
                [used_dict["returns"][i]    for i in idx],
                [used_dict["advantages"][i] for i in idx],
                [used_dict["values"][i]     for i in idx],
            )


# ============================================================
# PPOエージェント（ROSノード）— SB3スタイル命名/責務分離
# ============================================================
class PPOAgent(Node):
    def __init__(self, stage_num: str, max_training_episodes: str,load_prefix: str = ""):
        super().__init__('dqn_agent')  # ノード名は既存に合わせる

        # ---- 環境仕様 ----
        self.stage = int(stage_num)
        self.state_size = 26
        self.action_size = 2  # [linear, angular]
        self.max_training_episodes = int(max_training_episodes)
        self.load_prefix = load_prefix

        # ---- 行動の実レンジ（スケール後の最終レンジ）----
        self.lin_low, self.lin_high = -0.05, 0.30
        self.ang_low, self.ang_high = -1.5, 1.5

        # ---- PPOハイパーパラメータ（命名をSB3寄せ・値は完全据え置き）----
        self.gamma = 0.99
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.clip_range = 0.2            # ← clip_eps -> clip_range
        self.ent_coef = 0.01
        self.vf_coef = 1.0               # もともと value_loss に 0.5 を掛けているので、合成lossでは使わない（元実装維持）
        self.n_epochs = 10               # ← update_epochs -> n_epochs
        self.gae_lambda = 0.95

        # ★ ロールアウト長（2048ステップ貯めて更新）
        self.n_steps = 2048              # ← rollout_steps -> n_steps

        # ---- モデル初期化 ----
        self.device = torch.device('cpu')
        self.actor = Actor(self.state_size + 2, self.action_size).to(self.device)  
        self.critic = Critic(self.state_size + 2).to(self.device)  
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # ---- モデル保存パス ----
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        os.makedirs(self.model_dir_path, exist_ok=True)

        # 既存モデルのロード（継続学習 & カリキュラム学習）
        # ---- 手動指定チェックポイントの読み込み ----
        if self.load_prefix:
            actor_path = os.path.join(self.model_dir_path, f"{self.load_prefix}_actor.pth")
            critic_path = os.path.join(self.model_dir_path, f"{self.load_prefix}_critic.pth")

            if os.path.exists(actor_path) and os.path.exists(critic_path):
                try:
                    self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                    self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                    self.get_logger().info(
                        f"[Load] {self.load_prefix} からモデルをロードしました"
                    )
                    # 微調整っぽくしたいなら、ここで学習率を少し下げてもOK
                    # for g in self.opt_actor.param_groups: g['lr'] *= 0.5
                    # for g in self.opt_critic.param_groups: g['lr'] *= 0.5
                except Exception as e:
                    self.get_logger().error(f"[Load Error] {self.load_prefix} の読み込みに失敗: {e}")
            else:
                self.get_logger().warn(
                    f"[Load] 指定されたチェックポイント {self.load_prefix} が見つかりません。新規初期化で開始します。"
                )
        # ---- ROSインターフェース ----
        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, '/result', 10)
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(DqnSrv, 'reset_environment')
        self.rl_agent_interface_client = self.create_client(DqnSrv, 'rl_agent_interface')

        # ---- SB3風ロールアウトバッファ ----
        self.buffer = RolloutBuffer()

        # ---- トレーニング開始（SB3のlearn()に相当） ----
        self.learn()

    # ========================================================
    # 環境操作関連（元コードと同一）
    # ========================================================
    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for make_environment...')
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('reset_environment サービス待機中...')
        future = self.reset_environment_client.call_async(DqnSrv.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('環境リセット失敗: ゼロ状態で継続')
            state = np.zeros((1, self.state_size), np.float32)
        else:
            state = np.asarray(future.result().state, np.float32).reshape(1, self.state_size)

        if np.isnan(state).any():
            self.get_logger().warn('⚠️ 状態ベクトル内に NaN が検出されました')
        return state

    def step(self, action_pair):
        """環境に連続アクション[linear, angular]を送信"""
        req = DqnSrv.Request()
        req.action = [float(action_pair[0]), float(action_pair[1])]

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent_interface サービス待機中...')
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('rl_agent_interface 呼び出し失敗: 終了フラグを立てます')
            next_state = np.zeros((1, self.state_size), np.float32)
            reward, done = 0.0, True
        else:
            next_state = np.asarray(future.result().state, np.float32).reshape(1, self.state_size)
            reward = float(future.result().reward)
            done = bool(future.result().done)

        if np.isnan(next_state).any():
            self.get_logger().warn('⚠️ 次状態ベクトルに NaN を検出')
        return next_state, reward, done

    # ========================================================
    # ポリシー関連関数（元コードと同一）
    # ========================================================
    def _scale_actions(self, a_norm2: torch.Tensor) -> tuple:
        """
        a_norm2: shape (2,) in [-1, 1]^2 を実レンジにスケール
        """
        lin = (a_norm2[0].clamp(-1, 1).item() + 1.0) * 0.5 * (self.lin_high - self.lin_low) + self.lin_low
        ang = (a_norm2[1].clamp(-1, 1).item() + 1.0) * 0.5 * (self.ang_high - self.ang_low) + self.ang_low
        return (lin, ang)

    @staticmethod
    def _tanh_correction(raw_action: torch.Tensor) -> torch.Tensor:
        """
        tanh-squashのヤコビアン補正項:
        log|det(d tanh(u) / du)| = sum(log(1 - tanh(u)^2))
        数値安定な式: 2*(log(2) - u - softplus(-2u))
        """
        return 2.0 * (np.log(2.0) - raw_action - nn.functional.softplus(-2.0 * raw_action))

    def select_action(self, state_1xS):
        s = to_tensor(state_1xS, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)  # (1,S)

        with torch.no_grad():
            mu, std = self.actor(s)                 # (1,2), (1,2)
            dist = torch.distributions.Normal(mu, std)
            raw = dist.sample()                     # 未圧縮のガウスサンプル (1,2)
            a_norm = torch.tanh(raw).squeeze(0)     # [-1,1] に圧縮 (2,)

            # tanh補正込みのログ確率（保存用）
            logp_raw = dist.log_prob(raw).sum(dim=-1)           # (1,)
            log_det_j = self._tanh_correction(raw).sum(dim=-1)  # (1,)
            logp = (logp_raw - log_det_j).squeeze(0)            # スカラー

            value = self.critic(s).squeeze(0)                   # スカラー

        lin, ang = self._scale_actions(a_norm)
        return (lin, ang), float(value.item()), float(logp.item())

    # ========================================================
    # PPO更新ステップ（SB3の train() 相当）— 挙動は元の ppo_update() と完全一致
    # ========================================================
    def train(self, states, actions, old_logps, returns, advantages, old_values):
        # ---- to tensor on device ----
        states_t     = to_tensor(np.asarray(states,  np.float32), device=self.device)
        actions_t    = to_tensor(np.asarray(actions, np.float32), device=self.device)   # (T,2) in env-range
        old_logp_t   = to_tensor(np.asarray(old_logps, np.float32), device=self.device) # (T,)
        returns_t    = to_tensor(np.asarray(returns, np.float32), device=self.device)   # (T,)
        adv_t        = to_tensor(np.asarray(advantages, np.float32), device=self.device)
        old_values_t = to_tensor(np.asarray(old_values, np.float32), device=self.device)

        # 旧logpはtanh補正込みで記録している前提（select_actionで保存）
        adv_t = normalize_tensor(adv_t)

        # ---- tanh 補正付き log_prob 再計算 ----
        mu, std = self.actor(states_t)               # (T,2)
        dist = torch.distributions.Normal(mu, std)

        # 1) 環境スケール actions_t → [-1,1] に“逆スケール”
        lin = actions_t[:, 0]
        ang = actions_t[:, 1]
        lin_n = 2.0 * (lin - self.lin_low) / (self.lin_high - self.lin_low) - 1.0
        ang_n = 2.0 * (ang - self.ang_low) / (self.ang_high - self.ang_low) - 1.0
        a_norm = torch.stack([lin_n, ang_n], dim=-1).clamp(-0.999, 0.999)  # 数値安定

        # 2) [-1,1] → 生空間（tanhの前のu）へ戻す
        raw_actions = torch.atanh(a_norm)            # (T,2)

        # 3) tanh補正込みで logπ(a|s) を再計算
        logp_raw = dist.log_prob(raw_actions).sum(dim=-1)    # (T,)
        log_det_j = self._tanh_correction(raw_actions).sum(dim=-1)
        logp = logp_raw - log_det_j                          # (T,)

        ratio = torch.exp(logp - old_logp_t)                 # (T,)

        # ---- Actor Loss (clip) ----
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_t
        actor_loss = -torch.mean(torch.min(surr1, surr2)) - self.ent_coef * dist.entropy().mean()

        # ---- Critic Loss (value clipping) ----
        values = self.critic(states_t)                        # (T,)
        values_clipped = old_values_t + (values - old_values_t).clamp(-self.clip_range, self.clip_range)
        critic_loss_1 = (returns_t - values) ** 2
        critic_loss_2 = (returns_t - values_clipped) ** 2
        critic_loss = 0.5 * torch.mean(torch.max(critic_loss_1, critic_loss_2))

        # ---- Update ----
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.opt_actor.step()

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.opt_critic.step()

        entropy = dist.entropy().mean().item()
        return float(actor_loss.item()), float(critic_loss.item()), float(entropy)

    # ========================================================
    # 学習ループ（SB3の learn() 相当）— 挙動は元 process() と完全一致
    # ========================================================
    def learn(self):
        self.get_logger().info(
            f"=== PPO Continuous Training START06 {dt.month}/{dt.day} === "
        )

        self.env_make()
        time.sleep(1.0)

        episode_num = 0
        while episode_num < self.max_training_episodes:

            # エントロピー係数（元コードと同じ線形減衰）
            progress = episode_num / self.max_training_episodes
            self.ent_coef = 0.01 * (1.0 - progress)

            state = self.reset_environment()
            done = False
            prev_action = np.zeros((1, 2), np.float32)

            # エピソード内の一時バッファ
            ep_states, ep_actions, ep_rewards, ep_values, ep_logps = [], [], [], [], []
            total_reward = 0.0
            steps = 0

            while not done:
                steps += 1

                # ーーーーーー ネットワークへの入力値の確認（元コードそのまま） ーーーーーー
                # if steps % 20 == 0:
                #     goal_dist = state[0, 0]
                #     goal_angle = state[0, 1]
                #     self.get_logger().info(
                #         f"[NET INPUT] Step={steps:04d}, 距離={goal_dist:.3f} m, 角度={np.degrees(goal_angle):+.2f}°"
                #     )
                # ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

                state_plus_prev = np.concatenate([state, prev_action], axis=1)
                (lin, ang), value, logp = self.select_action(state_plus_prev)
                next_state, reward, done = self.step((lin, ang))

                total_reward += reward
                # GUI可視化：線・角・累積・即時
                gui = Float32MultiArray()
                gui.data = [float(lin), float(ang), float(total_reward), float(reward)]
                self.action_pub.publish(gui)

                # 一時バッファに記録（状態は1xS→S）
                ep_states.append(state_plus_prev.squeeze(0).tolist())
                ep_actions.append([float(lin), float(ang)])
                ep_rewards.append(float(reward))
                ep_values.append(float(value))
                ep_logps.append(float(logp))

                state = next_state
                prev_action = np.array([[lin, ang]], np.float32)
                time.sleep(0.020)
            
            final_value = self.critic(to_tensor(np.concatenate([next_state, prev_action], axis=1))).item()
            ep_values.append(final_value) # V(s_0)...V(s_{T-1}), V(s_T) の T+1個の値になる

            is_terminal_for_gae = (steps < 1500)
            # エピソード終了：GAEで Returns/Advantages を計算し、ロールアウトバッファに追記
            returns, advantages = compute_gae(ep_rewards, ep_values, is_terminal_for_gae, gamma=self.gamma, lam=self.gae_lambda)
            self.buffer.add_episode(ep_states, ep_actions, ep_logps, ep_values[:-1], returns, advantages)
            # ログ（毎エピソード）
            avg_v = float(np.mean(ep_values)) if ep_values else 0.0
            res = Float32MultiArray()
            res.data = [float(sum(ep_rewards)), float(avg_v)]
            self.result_pub.publish(res)

            episode_num += 1

            # ★ n_steps (2048) 以上たまったら更新（元コードと同一のトリガ）
            if self.buffer.size() >= self.n_steps:
                self.get_logger().info(
                    f"--- {self.buffer.size()}ステップ分のロールアウトで更新開始（{self.n_epochs} epochs） ---"
                )

                # 使う分（先頭 n_steps ステップ）を取り出し（元コード通り）
                used, extra_steps = self.buffer.take_head(self.n_steps)

                # === ミニバッチ更新（元コードのインデックスシャッフルと同じ挙動） ===
                batch_size = 256
                last_actor_loss, last_critic_loss, last_entropy = 0.0, 0.0, 0.0
                for epoch in range(self.n_epochs):
                    for batch_states, batch_actions, batch_logps, batch_returns, batch_advs, batch_values in \
                        self.buffer.iter_minibatches(used, batch_size):
                        last_actor_loss, last_critic_loss, last_entropy = self.train(
                            batch_states, batch_actions, batch_logps,
                            batch_returns, batch_advs, batch_values
                        )

                self.get_logger().info(
                    f"--- 更新完了 | ActorLoss={last_actor_loss:.3f}, "
                    f"CriticLoss={last_critic_loss:.3f}, Entropy={last_entropy:.3f} ---"
                )

            # モデル保存（据え置き）
            if episode_num % 50 == 0:
                torch.save(self.actor.state_dict(),
                           os.path.join(self.model_dir_path, f'stage{self.stage}_ep{episode_num}_actor.pth'))
                torch.save(self.critic.state_dict(),
                           os.path.join(self.model_dir_path, f'stage{self.stage}_ep{episode_num}_critic.pth'))
                with open(os.path.join(self.model_dir_path, f'stage{self.stage}_ep{episode_num}.json'), 'w') as f:
                    json.dump({"episode": episode_num}, f)

# ============================================================
# メイン関数（ノード起動部は据え置き）
# ============================================================
def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1500'
    load_prefix = args[3] if len(args) > 3 else ""

    rclpy.init(args=args)
    agent = PPOAgent(stage_num, max_training_episodes,load_prefix)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()