# RLNavigation_2025
事前地図を必要としない自律移動ロボットにおける深層強化学習の比較評価とカリキュラム学習の効果検証

# TurtleBot3 Reinforcement Learning Environment
*(ROS 2 Humble / Ubuntu 22.04)*

本リポジトリは、ROBOTIS 公式 eManual **TurtleBot3 Machine Learning** に基づき構築した、  
TurtleBot3 のシミュレーション環境における **強化学習（Reinforcement Learning）実行環境**である。

本 README では、**何も入っていない Ubuntu 22.04 環境から、学習実行まで**の手順を示す。

公式資料：  
https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/

---

## 1. 動作環境

- OS : Ubuntu 22.04 LTS  
- ROS : ROS 2 Humble Hawksbill  
- Python : 3.10  
- Simulator : Gazebo  
- Robot Model : TurtleBot3 (burger)

---

## 2. システム準備

    sudo apt update
    sudo apt -y upgrade
    sudo apt install -y software-properties-common curl git python3-pip
    sudo add-apt-repository universe
    sudo apt update

---

## 3. ROS 2 Humble インストール

Gazebo および RViz を含む desktop 版を使用する。

    sudo apt install -y ros-humble-desktop
    sudo apt install -y ros-dev-tools python3-colcon-common-extensions

ROS 環境を有効化する。

    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    source ~/.bashrc

---

## 4. TurtleBot3 ワークスペース構築

### 4.1 ワークスペース作成

    mkdir -p ~/turtlebot3_ws/src
    cd ~/turtlebot3_ws/src

### 4.2 TurtleBot3 関連パッケージ取得

    git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
    git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
    git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git
    git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

---

## 5. TurtleBot3 Machine Learning パッケージ取得

公式手順に従い、TurtleBot3 関連パッケージ取得後に Machine Learning を追加する。

    cd ~/turtlebot3_ws/src
    git clone -b humble-devel https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git

---

## 6. ビルド

    cd ~/turtlebot3_ws
    colcon build --symlink-install
    echo "source ~/turtlebot3_ws/install/setup.bash" >> ~/.bashrc
    source ~/.bashrc

---

## 7. Gazebo 関連パッケージ

    sudo apt install -y ros-humble-gazebo-*

---

## 8. 環境変数設定

    echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
    echo "export ROS_DOMAIN_ID=30" >> ~/.bashrc
    source ~/.bashrc

---

## 9. Python 依存パッケージ

    pip3 install --upgrade numpy==1.26.4 scipy==1.10.1 tensorflow==2.19.0 keras==3.9.2 pyqtgraph

---

## 10. シミュレーション起動確認

    ros2 launch turtlebot3_gazebo empty_world.launch.py

Gazebo 上に TurtleBot3 が表示されれば、環境構築は完了。

---

## 11. 強化学習（DQN）実行手順

### 11.1 学習用 Gazebo 環境起動（Stage 1）

    ros2 launch turtlebot3_gazebo turtlebot3_stage_1.launch.py

### 11.2 学習ノード起動

    ros2 launch turtlebot3_machine_learning turtlebot3_dqn_stage_1.launch.py

### 11.3 学習結果の可視化（任意）

    ros2 launch turtlebot3_machine_learning result_graph.launch.py

---

## 12. 備考

- 本リポジトリは **シミュレーション環境での学習実行**を目的とする  
- 実機適用時には別途 TurtleBot3 実機設定が必要  
- PPO 等の独自強化学習手法を導入する場合は、Python 仮想環境での管理を推奨する  

---

## 13. 参考資料

- ROBOTIS TurtleBot3 Machine Learning  
  https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/
