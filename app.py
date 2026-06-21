from flask import Flask, render_template, jsonify, request
from environment.grid_world import RobotNavEnv
from agent.nav_agent import get_agent
import threading
import time
import uuid

app = Flask(__name__)
app.secret_key = 'robot_nav_secret_2025'

# In-memory session store: {session_id: {'env':..., 'agent':..., 'running':..., ...}}
sessions = {}
sessions_lock = threading.Lock()


def make_session(algo='ppo', num_dynamic=4, num_static=6):
    env = RobotNavEnv(num_static_obstacles=num_static, num_dynamic_obstacles=num_dynamic)
    agent = get_agent(algo)
    return {
        'env': env,
        'agent': agent,
        'algo': algo,
        'episode': 1,
        'successes': 0,
        'collisions': 0,
        'timeouts': 0,
        'total_episodes': 0,
        'history': [],
        'done': False,
        'last_info': {},
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_session():
    data = request.get_json() or {}
    algo = data.get('algo', 'ppo')
    num_dynamic = int(data.get('num_dynamic', 4))
    num_static = int(data.get('num_static', 6))

    session_id = str(uuid.uuid4())
    with sessions_lock:
        sessions[session_id] = make_session(algo, num_dynamic, num_static)

    return jsonify({
        'session_id': session_id,
        'state': sessions[session_id]['env'].render_state()
    })


@app.route('/api/step', methods=['POST'])
def step():
    data = request.get_json() or {}
    session_id = data.get('session_id')

    with sessions_lock:
        sess = sessions.get(session_id)
        if not sess:
            return jsonify({'error': 'Invalid session'}), 404

        env = sess['env']
        agent = sess['agent']

        if sess['done']:
            return jsonify({
                'state': env.render_state(),
                'done': True,
                'info': sess['last_info'],
                'stats': get_stats(sess)
            })

        state = env._get_state()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        sess['last_info'] = info
        sess['done'] = done

        if done:
            sess['total_episodes'] += 1
            if info['goal_reached']:
                sess['successes'] += 1
            elif info['collision']:
                sess['collisions'] += 1
            elif info['timeout']:
                sess['timeouts'] += 1
            sess['history'].append({
                'episode': sess['total_episodes'],
                'result': 'success' if info['goal_reached'] else ('collision' if info['collision'] else 'timeout'),
                'steps': env.steps
            })

        return jsonify({
            'state': env.render_state(),
            'reward': round(reward, 2),
            'done': done,
            'info': info,
            'stats': get_stats(sess)
        })


@app.route('/api/reset_episode', methods=['POST'])
def reset_episode():
    data = request.get_json() or {}
    session_id = data.get('session_id')

    with sessions_lock:
        sess = sessions.get(session_id)
        if not sess:
            return jsonify({'error': 'Invalid session'}), 404

        sess['env'].reset()
        sess['done'] = False
        sess['last_info'] = {}

        return jsonify({'state': sess['env'].render_state(), 'stats': get_stats(sess)})


@app.route('/api/switch_algo', methods=['POST'])
def switch_algo():
    data = request.get_json() or {}
    session_id = data.get('session_id')
    new_algo = data.get('algo', 'ppo')

    with sessions_lock:
        sess = sessions.get(session_id)
        if not sess:
            return jsonify({'error': 'Invalid session'}), 404

        sess['agent'] = get_agent(new_algo)
        sess['algo'] = new_algo
        sess['env'].reset()
        sess['done'] = False
        # Keep cumulative stats across algo switches for comparison

        return jsonify({'state': sess['env'].render_state(), 'stats': get_stats(sess)})


def get_stats(sess):
    total = sess['total_episodes']
    success_rate = (sess['successes'] / total * 100) if total > 0 else 0
    collision_rate = (sess['collisions'] / total * 100) if total > 0 else 0
    return {
        'algo': sess['algo'],
        'total_episodes': total,
        'successes': sess['successes'],
        'collisions': sess['collisions'],
        'timeouts': sess['timeouts'],
        'success_rate': round(success_rate, 1),
        'collision_rate': round(collision_rate, 1),
        'history': sess['history'][-10:]
    }


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  AI-Based Autonomous Navigation System")
    print("  Deep Reinforcement Learning Robot Simulator")
    print("  Running at: http://127.0.0.1:5050")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5050)
