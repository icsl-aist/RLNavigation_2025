#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# # Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import os
import random
import subprocess
import sys
import time
import math

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Goal


ROS_DISTRO = os.environ.get('ROS_DISTRO')
if ROS_DISTRO == 'humble':
    from gazebo_msgs.srv import DeleteEntity
    from gazebo_msgs.srv import SpawnEntity
    from geometry_msgs.msg import Pose


class GazeboInterface(Node):

    def __init__(self, stage_num):
        super().__init__('gazebo_interface')
        self.stage = int(stage_num)

        self.entity_name = 'goal_box'
        self.entity_pose_x = 0.5
        self.entity_pose_y = 0.0

        # ★★★ 新規追加: 前回選択されたゴールインデックスを保持 ★★★
        self.last_goal_index = -1
        self._stage1_use_list1 = True 
        
        if ROS_DISTRO == 'humble':
            self.entity = None
            self.open_entity()
            self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
            self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
            self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.initialize_env_service = self.create_service(
            Goal,
            'initialize_env',
            self.initialize_env_callback,
            callback_group=self.callback_group
        )
        self.task_succeed_service = self.create_service(
            Goal,
            'task_succeed',
            self.task_succeed_callback,
            callback_group=self.callback_group
        )
        self.task_failed_service = self.create_service(
            Goal,
            'task_failed',
            self.task_failed_callback,
            callback_group=self.callback_group
        )

    def open_entity(self):
        try:
            package_share = get_package_share_directory('turtlebot3_gazebo')
            model_path = os.path.join(
                package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
            )
            with open(model_path, 'r') as f:
                self.entity = f.read()
            self.get_logger().info('Loaded entity from: ' + model_path)
        except Exception as e:
            self.get_logger().error('Failed to load entity file: {}'.format(e))
            raise e

    def spawn_entity(self):
        if ROS_DISTRO == 'humble':
            entity_pose = Pose()
            entity_pose.position.x = float(self.entity_pose_x)
            entity_pose.position.y = float(self.entity_pose_y)

            spawn_req = SpawnEntity.Request()
            spawn_req.name = self.entity_name
            spawn_req.xml = self.entity
            spawn_req.initial_pose = entity_pose

            while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('service for spawn_entity is not available, waiting ...')
            future = self.spawn_entity_client.call_async(spawn_req)
            rclpy.spin_until_future_complete(self, future)
            print(f'Spawn Goal at ({self.entity_pose_x}, {self.entity_pose_y}, {0.0})')
        else:
            service_name = '/world/dqn/create'
            package_share = get_package_share_directory('turtlebot3_gazebo')
            model_path = os.path.join(
                package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
            )
            req = (
                f'sdf_filename: "{model_path}", '
                f'name: "{self.entity_name}", '
                f'pose: {{ position: {{ '
                f'x: {self.entity_pose_x}, '
                f'y: {self.entity_pose_y}, '
                f'z: 0.0 }} }}'
            )
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1500',
                '--req', req
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                print(f'Spawn Goal at ({self.entity_pose_x}, {self.entity_pose_y}, {0.0})')
            except subprocess.CalledProcessError:
                pass

    def delete_entity(self):
        if ROS_DISTRO == 'humble':
            delete_req = DeleteEntity.Request()
            delete_req.name = self.entity_name

            while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('service for delete_entity is not available, waiting ...')
            future = self.delete_entity_client.call_async(delete_req)
            rclpy.spin_until_future_complete(self, future)
            print('Delete Goal')
        else:
            service_name = '/world/dqn/remove'
            req = f'name: "{self.entity_name}", type: 2'
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.Entity',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1500',
                '--req', req
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                print('Delete Goal')
            except subprocess.CalledProcessError:
                pass

    def reset_simulation(self):
        reset_req = Empty.Request()

        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for reset_simulation is not available, waiting ...')

        self.reset_simulation_client.call_async(reset_req)

    def reset_burger(self):
        service_name_delete = '/world/dqn/remove'
        req_delete = 'name: "burger", type: 2'
        cmd_delete = [
            'gz', 'service',
            '-s', service_name_delete,
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1500',
            '--req', req_delete
        ]
        try:
            subprocess.run(cmd_delete, check=True, stdout=subprocess.DEVNULL)
            print('Delete Burger')
        except subprocess.CalledProcessError:
            pass
        time.sleep(0.2)
        service_name_spawn = '/world/dqn/create'
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(package_share, 'models', 'turtlebot3_burger', 'model.sdf')
        req_spawn = (
            f'sdf_filename: "{model_path}", '
            f'name: "burger", '
            f'pose: {{ position: {{ x: 0.0, y: 0.0, z: 0.0 }} }}'
        )
        cmd_spawn = [
            'gz', 'service',
            '-s', service_name_spawn,
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1500',
            '--req', req_spawn
        ]
        try:
            subprocess.run(cmd_spawn, check=True, stdout=subprocess.DEVNULL)
            print('Spawn Burger')
        except subprocess.CalledProcessError:
            pass

    def task_succeed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = float(self.entity_pose_x)
        response.pose_y = float(self.entity_pose_y)
        response.success = True
        return response

    def task_failed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        if ROS_DISTRO == 'humble':
            self.reset_simulation()
        else:
            self.reset_burger()
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = float(self.entity_pose_x)
        response.pose_y = float(self.entity_pose_y)
        response.success = True
        return response

    def initialize_env_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        if ROS_DISTRO == 'humble':
            self.reset_simulation()
        else:
            self.reset_burger()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = float(self.entity_pose_x)
        response.pose_y = float(self.entity_pose_y)
        response.success = True
        return response

    def generate_goal_pose(self):

        if self.stage == 4:
            # --- 段階的に範囲拡大（最大到達後は最大で固定）---
            self._gen_count = getattr(self, "_gen_count", 0) + 1  # 呼ばれた回数を保持
            batch = 20          # 何回で更新
            start_units = 12     # 最初の距離　1/10 (1.0m)
            step_units  = 1     # 伸ばす距離　1/10 (0.1m)
            max_units   = 18    # 上限　1/10 (1.8m)

            level = (self._gen_count - 1) // batch
            limit = start_units + level * step_units
            if limit > max_units:
                limit = max_units  # 以降はずっと上限（サイクルしない）

            while True:
                # [-limit, +limit) を 0.1m刻みでサンプリング
                # ★ 修正: float()で明示的な型変換を追加
                self.entity_pose_x = float(random.randrange(-limit, limit) / 10.0)
                self.entity_pose_y = float(random.randrange(-limit, limit) / 10.0)
                break

        # 環境2におけるテスト座標
        elif self.stage == 1:
            # 5つの座標をすべて1つのリストにまとめます
            goal_pose_list = [
                [1.0, 1.0],
                [1.5, -1.5],
                [-1.5, 0],
                [-1.5, 1.5],
                [-1.5, -1.5]
            ]
            
            list_size = len(goal_pose_list)
            
            # 前回と同じインデックスが選ばれないようにループでチェック
            while True:
                rand_index = random.randint(0, list_size - 1)
                
                # self.last_goal_index は __init__ で -1 に初期化されている前提です
                if rand_index != self.last_goal_index:
                    break
            
            # 座標をセット (念のため float キャストを追加)
            self.entity_pose_x = float(goal_pose_list[rand_index][0])
            self.entity_pose_y = float(goal_pose_list[rand_index][1])
            
            # 今回選択したインデックスを保存
            self.last_goal_index = rand_index

        # 環境1におけるテスト座標
        elif self.stage == 4:
            # 5つの座標をすべて1つのリストにまとめます
            goal_pose_list = [
                [-1.0, 1.0],
                [1.0, 1.0],
                [0.0, -1.5],
                [1.5, -1.5],
                [-1.5, -1.5]
            ]
            
            list_size = len(goal_pose_list)
            
            # 前回と同じインデックスが選ばれないようにループでチェック
            while True:
                rand_index = random.randint(0, list_size - 1)
                
                # self.last_goal_index は __init__ で -1 に初期化されている前提です
                if rand_index != self.last_goal_index:
                    break
            
            # 座標をセット (念のため float キャストを追加)
            self.entity_pose_x = float(goal_pose_list[rand_index][0])
            self.entity_pose_y = float(goal_pose_list[rand_index][1])
            
            # 今回選択したインデックスを保存
            self.last_goal_index = rand_index

        # 環境１における学習座標
        elif self.stage == 4:
            # Stage 1: 2つのリストから交互にランダム選択
            
            # リスト1: 内側の4隅
            goal_pose_list_1 = [  
                [-1.0, 1.0],  
                [1.0, 1.0],  
                [0, -1.5],       
            ]
            
            # リスト2: 外側の4隅
            goal_pose_list_2 = [
                [1.5, -1.5],  
                [-1.5, -1.5],   
            ]
            
            if self._stage1_use_list1:
                current_list = goal_pose_list_1
                last_index_key = '_last_goal_index_list1'
            else:
                current_list = goal_pose_list_2
                last_index_key = '_last_goal_index_list2'
 
            list_size = len(current_list)

            last_index = getattr(self, last_index_key, -1)
            
            while True:
                rand_index = random.randint(0, list_size - 1)
                
                if rand_index != last_index:
                    break
 
            self.entity_pose_x = current_list[rand_index][0]
            self.entity_pose_y = current_list[rand_index][1]
            
            setattr(self, last_index_key, rand_index)

            self._stage1_use_list1 = not self._stage1_use_list1
 
            self.last_goal_index = (1 if self._stage1_use_list1 else 2) * 100 + rand_index

        elif self.stage == 4:
            # 公式ステージ4: 既存の固定リストから選択（変更なし）
            goal_pose_list = [
                [1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 1.5],
                [0.5, 2.0], [-1.5, 2.1], [-2.0, 0.5], [-2.0, -0.5],
                [-1.5, -2.0], [-0.5, -1.0], [2.0, -0.5], [-1.0, -1.0]
            ]
            rand_index = random.randint(0, len(goal_pose_list) - 1)
            # リストの値はfloatなので、キャスト不要
            self.entity_pose_x = goal_pose_list[rand_index][0]
            self.entity_pose_y = goal_pose_list[rand_index][1]

        elif self.stage == 4:
            # --- 新規ステージ: ランダム生成 + 壁際およびカスタム障害物際の除外 ---
            
            # 1. パラメータ定義
            # 部屋の境界線が不明なため、ランダム範囲を広く取り、除外で対応します
            MAX_COORD = 20 # ランダム生成範囲 [-2.0m, +2.0m] (通常 Gazebo の部屋サイズに合わせる)
            ROOM_BOUNDARY = 1.95 # 部屋の物理的な境界（マージン計算用）
            
            # 壁際からの最低安全距離 (0.25m マージンを確保)
            WALL_SAFETY_MARGIN = 0.25 
            
            # カスタム障害物の中心座標 (SDFファイルより)
            EXCLUDE_POINTS = [
                (-1.0, -1.5), # wall_1
                ( 0.0,  1.5), # wall_2
                ( 1.5, -1.0), # wall_3
            ]
            
            # 障害物からの最低安全半径 (0.80m を維持)
            EXCLUDE_RADIUS = 0.80 

            while True:
                # 2. ランダムな座標生成 (0.1m 刻み)
                x = random.randrange(-MAX_COORD, MAX_COORD) / 10.0
                y = random.randrange(-MAX_COORD, MAX_COORD) / 10.0
                
                # 3. 壁際からの除外チェック (0.25m マージンを確保)
                # 部屋の境界線を ROOM_BOUNDARY (1.95m) と仮定
                if not (abs(x) <= (ROOM_BOUNDARY - WALL_SAFETY_MARGIN) and
                        abs(y) <= (ROOM_BOUNDARY - WALL_SAFETY_MARGIN)):
                    continue # 壁に近すぎるため、再抽選

                # 4. 障害物からの除外チェック (0.80m)
                is_safe = True
                for ex_x, ex_y in EXCLUDE_POINTS:
                    # ゴール候補点 (x, y) から障害物中心までの距離
                    distance = math.hypot(x - ex_x, y - ex_y) 
                    
                    # 障害物自体の幅（0.05m〜1.0m）があるため、距離チェックのみでは不十分な場合があるが、
                    # ここでは中心からの安全半径 EXCLUDE_RADIUS を使用する
                    if distance < EXCLUDE_RADIUS:
                        is_safe = False
                        break
                
                if is_safe:
                    break # 全てのチェックを通過
            
            self.entity_pose_x = x
            self.entity_pose_y = y

        else:
            # その他のステージ (例: 1～3): 従来通りランダムにゴール生成
            # ★ 修正: float()で明示的な型変換を追加
            self.entity_pose_x = float(random.randrange(-18, 18) / 10)
            self.entity_pose_y = float(random.randrange(-18, 18) / 10)

def main(args=None):
    rclpy.init(args=sys.argv)
    stage_num = sys.argv[1] if len(sys.argv) > 1 else '1'
    gazebo_interface = GazeboInterface(stage_num)
    try:
        while rclpy.ok():
            rclpy.spin_once(gazebo_interface, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        gazebo_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



            # if self.stage == 1:
        #     # --- 段階的に範囲拡大（最大到達後は最大で固定）---
        #     self._gen_count = getattr(self, "_gen_count", 0) + 1  # 呼ばれた回数を保持
        #     batch = 40          # 何回で更新
        #     start_units = 10     # 最初の距離　1/10
        #     step_units  = 1     # 伸ばす距離　1/10
        #     max_units   = 18    # 上限　1/10

        #     level = (self._gen_count - 1) // batch
        #     limit = start_units + level * step_units
        #     if limit > max_units:
        #         limit = max_units  # 以降はずっと上限（サイクルしない）

        #     exclude1_x = -1.0
        #     exclude1_y = 0.0
        #     exclude2_x = 1.0
        #     exclude2_y = 0.0
        #     exclude3_x = 0.0
        #     exclude3_y = 1.0
        #     exclude4_x = 0.0
        #     exclude4_y = -1.0
        #     exclude_radius = 0.80

        #     while True:
        #         # [-limit, +limit) を 0.1m刻みでサンプリング
        #         self.entity_pose_x = random.randrange(-limit, limit) / 10.0
        #         self.entity_pose_y = random.randrange(-limit, limit) / 10.0
                
        #         # 除外エリアの中心からの距離を計算
        #         distance1 = math.sqrt(
        #             (self.entity_pose_x - exclude1_x)**2 + 
        #             (self.entity_pose_y - exclude1_y)**2
        #         )

        #         distance2 = math.sqrt(
        #             (self.entity_pose_x - exclude2_x)**2 + 
        #             (self.entity_pose_y - exclude2_y)**2
        #         )

        #         distance3 = math.sqrt(
        #             (self.entity_pose_x - exclude3_x)**2 + 
        #             (self.entity_pose_y - exclude3_y)**2
        #         )

        #         distance4 = math.sqrt(
        #             (self.entity_pose_x - exclude4_x)**2 + 
        #             (self.entity_pose_y - exclude4_y)**2
        #         )
                
        #         # 距離が半径(0.5m)以上なら、安全な座標なのでループを抜ける
        #         if distance1 >= exclude_radius and distance2 >= exclude_radius and \
        #            distance3 >= exclude_radius and distance4 >= exclude_radius:
        #             break

        
        # test用
        # elif self.stage == 4:
        #     # --- 修正: Stage 1 を完全ランダム + 障害物の形に合わせた長方形除外に変更 ---
            
        #     # 1. パラメータ定義
        #     MAX_COORD = 19 # ランダム生成範囲 [-1.9m, +1.9m] (0.1m 刻み)
        #     ROOM_BOUNDARY = 1.95 # 部屋の物理的な境界（マージン計算用）
        #     WALL_SAFETY_MARGIN = 0.20 # 壁際からの最低安全距離 (0.25m)
            
        #     # カスタム障害物の形状と除外マージンを定義
        #     # [中心x, 中心y, x方向マージン, y方向マージン]
        #     EXCLUDE_RECTS = [
        #         [-1.0, -1.5, 0.30, 0.80], # Wall 1: 縦長 (x:狭く, y:広く)
        #         [ 0.0,  1.5, 0.30, 0.80], # Wall 2: 縦長 (x:狭く, y:広く)
        #         [ 1.5, -1.0, 0.80, 0.30], # Wall 3: 横長 (x:広く, y:狭く)
        #     ]
            
        #     while True:
        #         # 2. ランダムな座標生成 (0.1m 刻み)
        #         x = random.randrange(-MAX_COORD, MAX_COORD) / 10.0
        #         y = random.randrange(-MAX_COORD, MAX_COORD) / 10.0
                
        #         # 3. 部屋の壁際からの除外チェック (0.25m マージンを確保)
        #         if not (abs(x) <= (ROOM_BOUNDARY - WALL_SAFETY_MARGIN) and
        #                 abs(y) <= (ROOM_BOUNDARY - WALL_SAFETY_MARGIN)):
        #             continue # 壁に近すぎるため、再抽選

        #         # 4. カスタム障害物からの長方形除外チェック
        #         is_safe = True
        #         for ex_x, ex_y, margin_x, margin_y in EXCLUDE_RECTS:
                    
        #             # 長方形の範囲を計算 (中心 ± マージン)
        #             x_min = ex_x - margin_x
        #             x_max = ex_x + margin_x
        #             y_min = ex_y - margin_y
        #             y_max = ex_y + margin_y
                    
        #             # 候補点 (x, y) が長方形内にいるかチェック
        #             # 長方形内 (除外対象) にいる条件:
        #             if (x_min <= x <= x_max) and (y_min <= y <= y_max):
        #                 is_safe = False
        #                 break
                
        #         if is_safe:
        #             break # 全てのチェックを通過
            
        #     self.entity_pose_x = x
        #     self.entity_pose_y = y
 
        #     # 従来の last_goal_index にはランダム生成フラグとして適当な値を格納
        #     self.last_goal_index = -999