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

Ubuntu 22.04 (Python 3.10) 環境における推奨バージョンをインストールする。
※ TensorFlow は 2.15.0 を使用（Keras は同梱版を使用するため個別インストール不要）。

    pip3 install --upgrade pip
    pip3 install numpy==1.26.4 tensorflow==2.15.0 pyqtgraph

---

## 10. シミュレーション起動確認

    ros2 launch turtlebot3_gazebo empty_world.launch.py

Gazebo 上に TurtleBot3 が表示されれば、環境構築は完了。
確認後、Ctrl+C で終了する。

---

## 11. 強化学習（DQN）実行手順

ROS 2 環境では、シミュレーション環境とエージェントノードの2つを起動して学習を行う。

### 1. ターミナル1：Gazeboシミュレーション起動
DQN用のステージ1を起動する。

    ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py

### 2. ターミナル2：DQNエージェントノード
強化学習アルゴリズムを実行するノードを起動し、学習を開始する。
（ROS 2 版パッケージでは、環境処理とエージェント処理が統合されている、あるいはトピック経由で連携するため、以下を実行するだけでよい）

    ros2 run turtlebot3_dqn dqn_agent

※ 学習の進捗グラフ（PyQtGraph）が表示され、ロボットが動き出せば成功。

---

## 12. 備考

- 本リポジトリは **シミュレーション環境での学習実行**を目的とする  
- 実機適用時には別途 TurtleBot3 実機設定が必要  
- PPO 等の独自強化学習手法を導入する場合は、Python 仮想環境（venv/conda）での管理を推奨する  

---

## 13. 参考資料

- ROBOTIS TurtleBot3 Machine Learning  
  https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/
