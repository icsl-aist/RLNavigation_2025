#!/usr/bin/env python3
#################################################################################
# PPO (Continuous Action) Test Node
# - Loads PPO Actor model (PyTorch)
# - Performs inference to get continuous action [linear_v, angular_v]
# - Communicates with RLEnvironment via 'rl_agent_interface' service
# - 🎯 100回実行して成功率を算出するよう変更
#################################################################################

import os
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torch.distributions

from turtlebot3_msgs.srv import Dqn as DqnSrv # DqnサービスをPPOのI/Fとして利用


# ============================================================
# PPO Actor モデルの定義（dqn_agent.py から移植）
# ============================================================
class Actor(nn.Module):
    """PPOエージェントのActorネットワーク（推論専用）"""
    def __init__(self, state_dim=28, action_dim=2):
        super().__init__()
        # state_dim=28 は、dqn_agent.py の state_size (26) + prev_action (2)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(128, action_dim)
        # log_std は学習済みパラメータとしてロードする
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.net(x)
        mu = torch.tanh(self.mu_layer(x))
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        return mu, std # mu:平均（アクションの中心）, std:標準偏差


# ============================================================
# PPO テストノード
# ============================================================
class PPOTest(Node):
    """TurtleBot3 PPOモデルのテスト実行ノード"""

    def __init__(self, stage, load_episode, num_tests=50):
        super().__init__('ppo_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)
        self.num_tests = int(num_tests) # 👈 実行回数 100回

        self.state_size = 26      # LiDAR + Goal Info
        self.action_size = 2      # [linear, angular]
        self.state_input_size = self.state_size + self.action_size # 28 (State + Prev Action)

        # ---- PPOパラメータ（dqn_agent.py からの移植）----
        self.lin_low, self.lin_high = -0.05, 0.30
        self.ang_low, self.ang_high = -1.5, 1.5
        self.device = torch.device('cpu')

        # ---- モデルの初期化とロード ----
        self.actor = Actor(self.state_input_size, self.action_size).to(self.device)
        self.load_model()

        # ---- ROSインターフェース ----
        self.rl_agent_interface_client = self.create_client(DqnSrv, 'rl_agent_interface')
        self.reset_env_client = self.create_client(DqnSrv, 'reset_environment')


        # ---- テスト実行開始 ----
        self.run_test()

    def load_model(self):
        """学習済みActorモデルをロード"""
        # モデルパスを PPO の命名規則に合わせる
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model',
            f'stage{self.stage}_ep{self.load_episode}_actor.pth'
        )

        try:
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval() # 推論モードに設定
            self.get_logger().info(f'🤖 Actorモデルをロードしました: {model_path}')
        except Exception as e:
            self.get_logger().error(f'❌ モデルのロードに失敗しました ({model_path}): {e}')
            sys.exit(1) # ロード失敗時は終了

    def _scale_actions(self, a_norm2: torch.Tensor) -> tuple:
        """
        [-1, 1]^2 の正規化アクションを実レンジにスケール（dqn_agent.pyから移植）
        """
        lin = (a_norm2[0].clamp(-1, 1).item() + 1.0) * 0.5 * (self.lin_high - self.lin_low) + self.lin_low
        ang = (a_norm2[1].clamp(-1, 1).item() + 1.0) * 0.5 * (self.ang_high - self.ang_low) + self.ang_low
        return (lin, ang)

    def get_action(self, state_plus_prev):
        """
        PPO Actorモデルを使用してアクションを選択（推論モード）
        """
        s = torch.from_numpy(state_plus_prev).to(device=self.device, dtype=torch.float32).unsqueeze(0) # (1, 28)

        with torch.no_grad():
            # 推論時は標準偏差を使わず、決定論的に平均値（μ）をアクションとする
            mu, _ = self.actor(s)               # (1,2), (1,2)
            a_norm = torch.tanh(mu).squeeze(0)  # [-1,1] に圧縮 (2,)

        return self._scale_actions(a_norm) # 実レンジ [lin, ang] に変換

    def reset_environment(self):
        """環境リセットサービスを呼び出し、初期状態を取得"""
        while not self.reset_env_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('reset_environment サービス待機中...')
        future = self.reset_env_client.call_async(DqnSrv.Request())
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is None:
            self.get_logger().error('環境リセット失敗: ゼロ状態で継続')
            state = np.zeros(self.state_size, np.float32)
        else:
            state = np.asarray(future.result().state, np.float32) # (26,)

        return state


    def run_test(self):
        self.get_logger().info(f'--- PPO Test START: {self.num_tests} エピソード実行 ---')
        
        episode_count = 0
        success_count = 0 # 👈 成功回数を記録

        # 👈 実行回数の上限を設定
        while rclpy.ok() and episode_count < self.num_tests:
            episode_count += 1
            
            # --- エピソード開始 ---
            state = self.reset_environment() # (26,)
            done = False
            succeed = False # 👈 成功フラグ
            score = 0.0
            local_step = 0
            
            # PPOでは、最初の行動を出す前に「前回の行動」prev_action = [0.0, 0.0] を結合する
            prev_action = np.zeros(self.action_size, np.float32) # (2,)

            self.get_logger().info(f'\n--- EPISODE {episode_count}/{self.num_tests} START ---')
            time.sleep(1.0)

            while not done:
                local_step += 1
                
                # 1. 状態に前回の行動を結合 (26+2=28)
                state_plus_prev = np.concatenate([state, prev_action], axis=0)

                # 2. Actorからアクション [linear_v, angular_v] を取得
                lin_v, ang_v = self.get_action(state_plus_prev)
                
                # 3. 環境サービスにアクションを送信
                req = DqnSrv.Request()
                req.action = [float(lin_v), float(ang_v)]

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn('rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    res = future.result()
                    state = np.asarray(res.state, np.float32) # 次の状態 (26,)
                    reward = res.reward
                    done = res.done
                    succeed = res.success # 👈 成功フラグを取得
                    score += reward
                    prev_action = np.array([lin_v, ang_v], np.float32) # 今回の行動を次ステップの 'prev_action' に
                    
                    if local_step % 50 == 0 or done:
                        self.get_logger().info(
                            f"[Step {local_step:04d}] Lin={lin_v:.3f}, Ang={ang_v:.3f} | "
                            f"R={reward:+.3f} | Total R={score:+.3f} | Done={done} (Success: {succeed})"
                        )
                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')
                    done = True # 失敗時は終了
                
                time.sleep(0.01)

            # 👈 成功回数を更新
            if succeed:
                success_count += 1
                result_str = "✅ SUCCESS"
            else:
                result_str = "❌ FAILED (Collision or Timeout)"
            
            self.get_logger().info(
                f'🏁 EPISODE {episode_count}/{self.num_tests} FINISHED in {local_step} steps. {result_str} | Total Score: {score:+.3f}'
            )

            # エピソード間に少し待機
            time.sleep(2.0)

        # ----------------------------------------------------
        # 最終結果の出力
        # ----------------------------------------------------
        success_rate = (success_count / self.num_tests) * 100.0
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("✨ テスト完了：成功率の算出結果")
        self.get_logger().info(f"総試行回数: {self.num_tests} エピソード")
        self.get_logger().info(f"成功回数: {success_count} 回")
        self.get_logger().info(f"成功率: {success_rate:.2f}%")
        self.get_logger().info("=" * 50)


def main(args=None):
    if args is None:
        args = sys.argv
        
    # 実行引数から Stage と Load Episode を取得
    stage = args[1] if len(args) > 1 else '1'
    load_episode = args[2] if len(args) > 2 else '600'
    num_tests = args[3] if len(args) > 3 else '100' # 👈 新しい引数

    rclpy.init(args=args)
    # 👈 PPOTest の初期化時に num_tests を渡す
    node = PPOTest(stage, load_episode, num_tests)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()