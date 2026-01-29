#!/usr/bin/env python3
#################################################################################
# PPO (Continuous Action) version of TurtleBot3 RL Environment
# - Compatible with ROS2 Humble
# - Supports continuous angular velocity control (float)
# - Keeps service/topic interfaces for backward compatibility
#################################################################################

import math
import time
import os
import numpy
import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn, Goal


ROS_DISTRO = os.environ.get('ROS_DISTRO')  # 実行環境が "humble" か判定用


class RLEnvironment(Node):
    """TurtleBot3用 強化学習環境ノード（PPO連続アクション対応版）"""

    def __init__(self):
        # ROS 2ノードを作成
        super().__init__('rl_environment')

        # ---- ゴールおよびロボット位置状態 ----
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0
        self.goal_angle = 0.0             # ロボットから見たゴール方向角
        self.goal_distance = 1.0          # 現在のゴール距離
        self.init_goal_distance = 0.5     # 初期距離（リセット時）

        # ---- ステップ管理／エピソード制御 ----
        self.max_step = 1500               # 1エピソードあたり最大ステップ数
        self.local_step = 0               # 現在のステップ数
        self.done = False                 # エピソード終了フラグ
        self.fail = False                 # 失敗フラグ（衝突など）
        self.succeed = False              # 成功フラグ（ゴール到達）
        self.stop_cmd_vel_timer = None    # 自動停止用タイマー

        self._ep_steps = 0
        self.episode_count = 0            # エピソードカウントを追加
        self.trajectory_log = []          # 軌跡ログリストを追加

        # ---- 報酬コンポーネント保持用 (★ 追加: get_reward_componentsで必要) ----
        self.distance_reward = 0.0
        self.yaw_reward = 0.0
        self.obstacle_reward = 0.0
        self.terminal_reward = 0.0

        # ---- LiDAR（LaserScan）情報 ----
        self.scan_ranges = []             # 全方向距離
        self.front_ranges = []            # 前方距離（0〜90°, 270〜360°）
        self.front_angles = []            # 前方角度群
        self.min_obstacle_distance = 10.0 # 最近障害物距離


        # ---- QoS / Publisher設定 ----
        qos = QoSProfile(depth=10)
        # ROS2ディストロによりTwist/TwistStampedを切り替え
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)


        # ---- サブスクライブ：オドメトリとLiDAR ----
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_sub_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_sub_callback, qos_profile_sensor_data)

        # ---- 環境制御サービスのクライアント群 ----
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(Goal, 'task_succeed', callback_group=self.clients_callback_group)
        self.task_failed_client = self.create_client(Goal, 'task_failed', callback_group=self.clients_callback_group)
        self.initialize_environment_client = self.create_client(Goal, 'initialize_env', callback_group=self.clients_callback_group)

        # ---- 強化学習ノードとの通信サービス（サーバ）----
        self.rl_agent_interface_service = self.create_service(Dqn, 'rl_agent_interface', self.rl_agent_interface_callback)
        self.make_environment_service = self.create_service(Empty, 'make_environment', self.make_environment_callback)
        self.reset_environment_service = self.create_service(Dqn, 'reset_environment', self.reset_environment_callback)

    # ============================================================================
    # 環境初期化／リセット関連
    # ============================================================================
    def make_environment_callback(self, request, response):
        """環境生成（ゴール位置初期化をトリガー）"""
        self.get_logger().info('Make environment called')
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('initialize_env service not available, waiting...')
        # initialize_envを呼び出してゴール位置を取得
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal or not response_goal.success:
            self.get_logger().error('initialize_env request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(f'Goal initialized at [{self.goal_pose_x:.2f}, {self.goal_pose_y:.2f}]')
        return response

    # この関数は新しいエピソードが開始するたびに呼び出される
    # ロボットの現在の状態を観測し、それを初期状態としてエージェントに返します
    def reset_environment_callback(self, request, response):
        """環境リセット時の状態ベクトル生成"""
        state = self.calculate_state()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        response.state = state
        return response

    # ============================================================================
    # Gazebo環境との連携サービス（ゴール更新など）
    # ============================================================================
    # エピソードが終了した際にこのノード自身が内部的に呼び出すメソッド

    # ゴールした場合
    def call_task_succeed(self):
        """ゴール到達時に呼び出し：次のゴールを生成"""
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for task_succeed...')
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result:
            self.goal_pose_x, self.goal_pose_y = result.pose_x, result.pose_y
            self.get_logger().info('Task succeed service finished')

    # 障害物に衝突または時間切れになった場合
    def call_task_failed(self):
        """衝突／タイムアウト時に呼び出し：ゴールを再配置"""
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for task_failed...')
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result:
            self.goal_pose_x, self.goal_pose_y = result.pose_x, result.pose_y
            self.get_logger().info('Task failed service finished')

    # ============================================================================
    # センサコールバック（LiDAR + Odometry）
    # ============================================================================
    # ロボットのセンサーからデータが送られてくるたびに実行され、生のデータを強化学習で使える形式の情報に変換
    def scan_sub_callback(self, scan):
        """LiDARスキャンを受信し、360°を24分割して代表距離を抽出"""
        # --- 1. LaserScan → NumPy配列 ---
        raw_ranges = np.array(scan.ranges, dtype=np.float32)
        raw_ranges[np.isinf(raw_ranges)] = 3.5     # Infを最大距離に置換
        raw_ranges[np.isnan(raw_ranges)] = 1.5     # NaNを0に置換

        num_rays = len(raw_ranges)
        num_bins = 24                              # ★ 24分割
        step = num_rays // num_bins                # 各セクタの幅（例: 360/24=15）

        # --- 2. 各15°セクタの代表距離（最小値）を抽出 ---
        sector_mins = []
        for i in range(num_bins):
            start = i * step
            end = (i + 1) * step
            sector_slice = raw_ranges[start:end]
            min_dist = float(np.min(sector_slice))  # 障害物に敏感な「最小値」を採用
            sector_mins.append(min_dist)

        # --- 3. 結果を保存 ---
        self.scan_ranges = sector_mins              # ← 24要素固定
        self.min_obstacle_distance = min(sector_mins)

    def odom_sub_callback(self, msg):
        """オドメトリ情報を受信し、ロボット姿勢とゴール相対角を更新"""
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        # ゴールまでの距離と角度を計算
        goal_distance = math.hypot(self.goal_pose_x - self.robot_pose_x, self.goal_pose_y - self.robot_pose_y)
        # 
        path_theta = math.atan2(self.goal_pose_y - self.robot_pose_y, self.goal_pose_x - self.robot_pose_x)
        goal_angle = path_theta - self.robot_pose_theta

        # [-π, π]範囲に正規化
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    # ============================================================================
    # 状態ベクトルおよび報酬設計
    # ============================================================================
    def calculate_state(self):
        """ロボットの状態（距離・角度＋360°→24分割LiDAR）をベクトル化"""
        state = [float(self.goal_distance), float(self.goal_angle)]

        # ★ 前方のみでなく360°分割値を使用
        for var in self.scan_ranges:
            state.append(float(var))

        # ---- 以下は既存の終了条件チェックをそのまま維持 ----
        if self.goal_distance < 0.20:
            self.get_logger().info('Goal Reached')
            self.succeed, self.done = True, True
            self.publish_stop()
            self.local_step = 0
            self.call_task_succeed()

        if self.min_obstacle_distance < 0.15:
            self.get_logger().info('Collision')
            self.fail, self.done = True, True
            self.publish_stop()
            self.local_step = 0
            self.call_task_failed()

        if self.local_step >= self.max_step:
            self.get_logger().info('Time Out')
            self.done = True          # ★ 修正: done フラグのみ True にする
            # self.fail = True        # ★ 削除: タイムアウト時は self.fail を立てない
            self.publish_stop()
            self.local_step = 0
            self.call_task_failed()

        return state


    # def compute_directional_weights(self, relative_angles, max_weight=10.0):
    #     """前方方向に強く重みをかける角度依存関数"""
    #     power = 6
    #     raw = (numpy.cos(relative_angles)) ** power + 0.1
    #     scaled = raw * (max_weight / numpy.max(raw))
    #     return scaled / numpy.sum(scaled)

    # def compute_weighted_obstacle_reward(self):
    #     """LiDAR距離に基づく障害物ペナルティ（前方集中型）"""
    #     if not self.front_ranges or not self.front_angles:
    #         return 0.0
    #     front_ranges = numpy.array(self.front_ranges)
    #     front_angles = numpy.array(self.front_angles)
    #     mask = front_ranges <= 0.4
    #     if not numpy.any(mask):
    #         return 0.0
    #     front_ranges = front_ranges[mask]
    #     front_angles = front_angles[mask]
    #     rel_angles = numpy.unwrap(front_angles)
    #     rel_angles[rel_angles > numpy.pi] -= 2 * numpy.pi
    #     weights = self.compute_directional_weights(rel_angles)
    #     safe = numpy.clip(front_ranges - 0.20, 1e-2, 3.5)
    #     decay = numpy.exp(-2.0 * safe)
    #     weighted = numpy.dot(weights, decay)
    #     return - (0.8 + 2.5 * weighted)

    # # Lidar最短距離一本だけをペナルティとする
    # def compute_weighted_obstacle_reward(self):
        
    #     if not self.scan_ranges:
    #         return 0.0
        
    #     ranges_360 = numpy.array(self.scan_ranges)

    #     mask = ranges_360 <= 0.4
    #     if not numpy.any(mask):
    #         return 0.0
        
    #     close_ranges = ranges_360[mask]
    #     safe = numpy.clip(close_ranges - 0.20, 1e-2, 3.5) #
    #     decay = numpy.exp(-2.0 * safe) #

    #     weighted = numpy.sum(decay)
        
    #     return - (10 + 2.5 * weighted)

    # すべてのLidarにペナルティ（一定）
    # def compute_weighted_obstacle_reward(self):

    #     if not self.scan_ranges:
    #         return 0.0
        
    #     ranges = np.array(self.scan_ranges)

    #     # --- 設定パラメータ ---
    #     threshold_dist = 0.30     
    #     fixed_penalty = 0.8       
    #     is_close = np.any(ranges < threshold_dist)

    #     if is_close:
    #         return -fixed_penalty
    #     else:
    #         return 0.0

    # すべてのLidarにペナルティ（比例）
    def compute_weighted_obstacle_reward(self):
 
        if not self.scan_ranges:
            return 0.0

        # NumPy配列に変換
        ranges = np.array(self.scan_ranges)
 
        # 👈 修正: しきい値を 0.30m に固定
        threshold_dist = 0.35  

        # 👈 修正: 係数を 0.081 に設定 (0.15mでペナルティが 3.00 になるように調整)
        penalty_scale  = 0.081
 
        mask = ranges < threshold_dist
        if not np.any(mask):
            return 0.0
 
        close_points = ranges[mask]
 
        # 0除算を防ぐため、非常に小さい値 (e.g., 1e-6) を距離に加算

        # 1. 逆数の差分を計算: (1 / 距離) - (1 / しきい値(0.30))
        diffs_inverse = (1.0 / (close_points + 1e-6)) - (1.0 / threshold_dist)

        # 2. その差分を三乗する (増加率を急激にする)
        # ★★★ power=3 を採用 ★★★
        penalty_component = diffs_inverse ** 3

        total_penalty = np.sum(penalty_component) * penalty_scale
        return -total_penalty
 

    def calculate_reward(self):
        """報酬の計算とログ出力（修正）"""
        
        # 1. ゴール距離の計算
        distance_to_goal = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2 +
            (self.goal_pose_y - self.robot_pose_y) ** 2
        )

        # 初回のみ初期化
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = distance_to_goal

        # 2. 距離報酬の計算
                # 2. 距離報酬の計算
        if self.local_step == 0:
            distance_diff = 0.0
        else:
            distance_diff = self.prev_distance - distance_to_goal
            # self.distance_reward = distance_diff * 300.0
            self.distance_reward = max(0.0, distance_diff) * 100.0

        # 3. 角度報酬の計算
        self.yaw_reward = 0.0 * math.cos(self.goal_angle) # ★ インスタンス変数に保存

        # 4. 障害物ペナルティの計算
        self.obstacle_reward = self.compute_weighted_obstacle_reward() # ★ インスタンス変数に保存

        # 5. 終了時報酬の計算
        steps = max(1, getattr(self, "_ep_steps", self.local_step))
        S = float(self.max_step)
        self.terminal_reward = 0.0 # ★ インスタンス変数に保存
        if self.succeed:
            succ_scale = max(0.2, 1.0 - (steps - 1) / S) 
            self.terminal_reward = 600.0 * succ_scale
        elif self.fail:
            fail_scale = min(1.5, steps / S)
            self.terminal_reward = -500.0 * fail_scale  

        # 6. 総報酬計算
        reward = self.distance_reward + self.yaw_reward + self.obstacle_reward + self.terminal_reward

        # --- 7. 前回距離の更新 ---
        self.prev_distance = distance_to_goal

        return reward
    
    # ★ エラー解消のための追加メソッド ★
    def get_reward_components(self):
        """計算された報酬コンポーネントを辞書で返すヘルパー関数"""
        return {
            'distance_reward': self.distance_reward,
            'yaw_reward': self.yaw_reward,
            'obstacle_reward': self.obstacle_reward,
            'terminal_reward': self.terminal_reward,
        }

    # ============================================================================
    # 連続アクションのコールバック（エージェント→環境）
    # ============================================================================
    def rl_agent_interface_callback(self, request, response):
        """エージェント（PPO）から連続アクションを受け取り実行"""

        self._ep_steps = getattr(self, "_ep_steps", 0) + 1
        try:
            linear_v = float(np.clip(request.action[0], -0.06, 0.30))
            angular_v = float(np.clip(request.action[1], -1.5,  1.5))
        except Exception:
            linear_v = 0.0
            angular_v = 0.0

        # --- デッドバンド（微小な誤差で動かないように）
        if abs(linear_v) < 0.005:
            linear_v = 0.0


        # --- Twist 発行 ---
        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = linear_v
            msg.angular.z = angular_v
        else:
            msg = TwistStamped()
            msg.twist.linear.x = linear_v
            msg.twist.angular.z = angular_v

        self.cmd_vel_pub.publish(msg)
        # self.restart_stop_timer()

        # --- 状態と報酬 ---
        response.state = self.calculate_state()
        response.reward = self.calculate_reward() 
        response.done = self.done
        response.success = self.succeed

        self.local_step += 1
        reward_info = self.get_reward_components() # 報酬コンポーネントを取得 (★ エラー解消)

        # -------------------------------
        # 🧭 1ステップごとの軌跡ログを記録
        # -------------------------------
        if not hasattr(self, 'trajectory_log') or self.local_step == 1:
            # self.trajectory_logは__init__で初期化済みだが、念のためepisode_countもチェック
            self.episode_count = getattr(self, 'episode_count', 0)

        self.trajectory_log.append((
            self.episode_count + 1,        
            self.local_step,
            round(self.robot_pose_x, 3),
            round(self.robot_pose_y, 3),
            round(linear_v, 3),
            round(angular_v, 3),
            round(reward_info['distance_reward'], 3),
            round(reward_info['yaw_reward'], 3),
            round(reward_info['obstacle_reward'], 3),
            round(reward_info['terminal_reward'], 3),
            round(response.reward, 3)
        ))

        # -------------------------------
        # 🎯 エピソード終了時：ファイルに保存
        # -------------------------------
        if self.done:
            final_step = self.local_step

            self._ep_steps = 0
            self.episode_count += 1
            self.get_logger().info(f"🏁 エピソード {self.episode_count} 終了")
            self.get_logger().info("ーーーーーーーーーーーーーーーーーーーーーー")
            self.get_logger().info(f"終了ステップ数: {final_step}")
            self.get_logger().info("ーーーーーーーーーーーーーーーーーーーーーー")

            # 30エピソードごとに保存
            if self.episode_count % 5 == 0:
                # 保存ディレクトリの定義と作成
                save_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                    'saved_model'
                )
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                start_ep = self.episode_count - 4 
                end_ep = self.episode_count
                save_path = os.path.join(save_dir, f"trajectory_ep{start_ep}_to_{end_ep}.csv")
                
                # CSVファイルの内容を整形してログ出力
                self.get_logger().info("\n\n" + "="*112)
                self.get_logger().info(f"💾 軌跡ログ保存 ({start_ep}〜{end_ep})")
                self.get_logger().info("==================================================================================================================")
                
                # 日本語ヘッダー
                header = "| エピソード | Step |   X座標  |   Y座標  | 線形速度 | 角速度 | 距離報酬 | 角度報酬 | 障害物報酬 | 終了時報酬 | 総報酬 |"
                self.get_logger().info(header)
                # 区切り線
                self.get_logger().info("------------------------------------------------------------------------------------------------------------------")
                
                with open(save_path, "w") as f:
                    # ヘッダーは解析のしやすさのため英語変数名を維持
                    f.write("episode,step,x,y,linear,angular,distance_reward,yaw_reward,obstacle_reward,terminal_reward,total_reward\n")
                    
                    # 修正箇所: タプルの要素を日本語の変数名に割り当ててCSVに書き出し
                    for (エピソード, ステップ, X座標, Y座標, 線形速度, 角速度, 距離報酬, 角度報酬, 障害物報酬, 終了時報酬, 総報酬) in self.trajectory_log:
                        # CSV書き出し (値はカンマ区切りで書き出し)
                        f.write(f"{エピソード},{ステップ},{X座標},{Y座標},{線形速度},{角速度},{距離報酬},{角度報酬},{障害物報酬},{終了時報酬},{総報酬}\n")
                        
                        # ログ出力も整形して表示
                        self.get_logger().info(
                            f"| {エピソード:^10d} | {ステップ:^4d} | {X座標:^8.4f} | {Y座標:^8.4f} | {線形速度:^8.3f} | {角速度:^6.3f} | {距離報酬:^8.3f} | {角度報酬:^8.3f} | {障害物報酬:^10.3f} | {終了時報酬:^10.3f} | {総報酬:^6.3f} |"
                        )
                
                self.get_logger().info("==================================================================================================================")
                self.get_logger().info(f"💾 30エピソード分の軌跡を保存しました: {save_path}")
                self.trajectory_log = []  # ログをリセット

            # 状態リセット
            self.publish_stop()

            time.sleep(0.3)
            self.done = False
            self.succeed = False
            self.fail = False
            self.local_step = 0

        self.last_cmd_lin = linear_v
        self.last_cmd_ang = angular_v


        return response

    def restart_stop_timer(self):
        """1ステップごとに短時間後ロボットを停止するタイマーを設定"""
        if self.stop_cmd_vel_timer is not None:
            self.destroy_timer(self.stop_cmd_vel_timer)
        self.stop_cmd_vel_timer = self.create_timer(0.2, self.timer_callback)

    def timer_callback(self):
        """タイマー到達時に停止コマンドを発行"""
        self.publish_stop()
        self.destroy_timer(self.stop_cmd_vel_timer)

    def publish_stop(self):
        """ロボット停止（cmd_vel=0）をパブリッシュ"""
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())

    # ============================================================================
    # Utility関数
    # ============================================================================
    def euler_from_quaternion(self, quat):
        """四元数 → オイラー角（roll, pitch, yaw）変換"""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw


# ============================================================================
# ROSノードエントリポイント
# ============================================================================
def main(args=None):
    rclpy.init(args=args)
    env = RLEnvironment()
    try:
        # メインループ：ROSスピンでサービスと購読を処理
        while rclpy.ok():
            rclpy.spin_once(env, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()