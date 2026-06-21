# 🤖 NavSim – AI-Based Autonomous Navigation System for Mobile Robots

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![RL](https://img.shields.io/badge/Algorithm-DQN%2FPPO-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

> A Final Year B.Voc AI&DS Project — Anjuman-I-Islam's Abdul Razzak Kalsekar Polytechnic, New Panvel

---

## 📌 About

**NavSim** is a complete, working web application that simulates a mobile robot navigating to a goal while avoiding static and dynamic obstacles — entirely in your browser, no ROS or Gazebo installation required.

It demonstrates and visually compares three navigation strategies:

| Agent | Description |
|---|---|
| 🟢 **PPO** | Proximal Policy Optimization — stable, smooth navigation policy |
| 🟡 **DQN** | Deep Q-Network — reactive policy, slightly less stable |
| 🔵 **A\*** | Classical path planning — fails in dynamic environments |

---

## 🚀 Features

- 🗺️ **Live Canvas Visualization** — Watch the robot navigate in real time
- 🎛️ **Adjustable Environment** — Control number of static/dynamic obstacles
- 🔄 **Algorithm Switching** — Compare PPO vs DQN vs A* on the fly
- 📊 **Live Statistics** — Success rate, collision rate, episode history
- 🧠 **Real RL Training Script Included** — `train_agent.py` uses actual Stable-Baselines3 PPO/DQN
- ⚡ **Instant Demo** — No GPU, no ROS, no Gazebo needed to run the web app

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, Flask 3.0 |
| Simulation | Custom 2D physics engine (`environment/grid_world.py`) |
| RL Training (optional) | Stable-Baselines3 (PPO, DQN) |
| Frontend | HTML5 Canvas, Bootstrap 5, JavaScript |

---

## ⚙️ Setup & Run (Web Demo)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/robot-navigation-drl.git
cd robot-navigation-drl
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
```
http://127.0.0.1:5050
```

That's it — no ROS, no Gazebo, no GPU needed. The simulation runs instantly!

---

## 🎓 Training Real RL Models (Optional, Advanced)

The web demo uses lightweight reactive policies that mimic converged PPO/DQN
behavior so it runs instantly. If you want to train **actual** Stable-Baselines3
models from scratch on the same environment:

```bash
pip install stable-baselines3 gymnasium torch

python train_agent.py --algo ppo --timesteps 500000
python train_agent.py --algo dqn --timesteps 500000
```

This trains a real RL agent and saves a `.zip` model checkpoint to `models/`,
plus prints evaluation metrics (success rate, collision rate) over 100 test episodes.

---

## 📁 Project Structure

```
robot-navigation-drl/
├── app.py                      # Main Flask application
├── train_agent.py              # Real Stable-Baselines3 training script
├── requirements.txt
├── README.md
├── environment/
│   ├── __init__.py
│   └── grid_world.py           # Custom 2D navigation environment (LIDAR sim)
├── agent/
│   ├── __init__.py
│   └── nav_agent.py            # PPO / DQN / A* agent policies
├── templates/
│   └── index.html              # Main web page
├── static/
│   ├── css/style.css
│   └── js/main.js              # Canvas rendering + simulation loop
└── models/                     # Trained model checkpoints (generated)
```

---

## 🧠 How It Works

1. **Environment** (`grid_world.py`) simulates a 2D world with a robot, goal,
   and obstacles. The robot has a simulated 24-ray LIDAR sensor.
2. **State** is a 26-dimensional vector: 24 LIDAR distances + goal distance + goal angle.
3. **Agent** (`nav_agent.py`) selects one of 5 actions (forward, slight/sharp left/right)
   based on the state, following PPO-style or DQN-style behavior.
4. **Flask backend** (`app.py`) runs the simulation step-by-step and streams
   state to the browser via JSON API calls.
5. **Frontend** (`main.js`) renders the robot, obstacles, goal and trajectory
   on an HTML5 Canvas in real time.

---

## 📊 Reported Results (from Black Book / Research Paper)

| Algorithm | Simple Env | Medium Env | Complex (Dynamic) Env |
|---|---|---|---|
| A* (Baseline) | 98.2% | — | 61.2% |
| DQN | 94.1% | 88.5% | 83.7% |
| **PPO (Best)** | 96.3% | 93.8% | **91.4%** |

---

## 👨‍💻 Developed By

**[Your Name]**
B.Voc in AI & Data Science — Final Year
Anjuman-I-Islam's Abdul Razzak Kalsekar Polytechnic, New Panvel
Under the guidance of **Mr. Ali Karim**

---

## 📄 License

MIT License — Free to use for educational purposes.
