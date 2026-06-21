// ── Global State ──
let sessionId = null;
let isPlaying = false;
let simInterval = null;
let canvas, ctx;
let worldSize = 10;
let lastState = null;

document.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('simCanvas');
  ctx = canvas.getContext('2d');

  document.getElementById('staticRange').addEventListener('input', (e) => {
    document.getElementById('staticVal').textContent = e.target.value;
  });
  document.getElementById('dynamicRange').addEventListener('input', (e) => {
    document.getElementById('dynamicVal').textContent = e.target.value;
  });

  drawEmptyState();
});

function drawEmptyState() {
  ctx.fillStyle = '#1a2332';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#445';
  ctx.font = '20px Poppins';
  ctx.textAlign = 'center';
  ctx.fillText('Click "Start New Simulation" to begin', canvas.width / 2, canvas.height / 2);
}

// ── Start Simulation ──
async function startSimulation() {
  const algo = document.getElementById('algoSelect').value;
  const numStatic = document.getElementById('staticRange').value;
  const numDynamic = document.getElementById('dynamicRange').value;

  const res = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algo, num_static: numStatic, num_dynamic: numDynamic })
  });
  const data = await res.json();
  sessionId = data.session_id;
  worldSize = data.state.world_size;
  lastState = data.state;

  drawState(data.state);
  updateStatus('Running', 'success');
  document.getElementById('playPauseBtn').disabled = false;
  document.getElementById('resetBtn').disabled = false;
  document.getElementById('infoBanner').innerHTML =
    `🤖 <strong>${algo.toUpperCase()}</strong> agent is navigating to the goal while avoiding obstacles...`;

  isPlaying = true;
  document.getElementById('playPauseBtn').innerHTML = '⏸️ Pause';
  startLoop();
}

// ── Simulation Loop ──
function startLoop() {
  if (simInterval) clearInterval(simInterval);
  const speed = parseInt(document.getElementById('speedRange').value);
  const delay = Math.max(20, 220 - speed * 20);

  simInterval = setInterval(async () => {
    if (!isPlaying || !sessionId) return;
    await stepSimulation();
  }, delay);
}

async function stepSimulation() {
  const res = await fetch('/api/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId })
  });
  const data = await res.json();

  lastState = data.state;
  drawState(data.state);
  updateStats(data.stats);

  if (data.reward !== undefined) {
    document.getElementById('statReward').textContent = data.reward;
  }

  if (data.done) {
    handleEpisodeEnd(data.info);
    // Auto-start next episode after brief pause
    setTimeout(async () => {
      if (isPlaying) {
        await fetch('/api/reset_episode', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId })
        }).then(r => r.json()).then(d => {
          lastState = d.state;
          drawState(d.state);
          updateStats(d.stats);
        });
      }
    }, 800);
  }
}

function handleEpisodeEnd(info) {
  if (info.goal_reached) {
    updateStatus('Goal Reached! 🎉', 'success');
  } else if (info.collision) {
    updateStatus('Collision! 💥', 'danger');
  } else if (info.timeout) {
    updateStatus('Timeout ⏱️', 'warning');
  }
}

function updateStatus(text, type) {
  const badge = document.getElementById('statusBadge');
  badge.textContent = text;
  badge.className = `badge bg-${type}`;
}

// ── Toggle Play/Pause ──
function togglePlay() {
  isPlaying = !isPlaying;
  const btn = document.getElementById('playPauseBtn');
  btn.innerHTML = isPlaying ? '⏸️ Pause' : '▶️ Resume';
  if (isPlaying) {
    updateStatus('Running', 'success');
    startLoop();
  } else {
    updateStatus('Paused', 'secondary');
    if (simInterval) clearInterval(simInterval);
  }
}

// ── Reset Episode ──
async function resetEpisode() {
  if (!sessionId) return;
  const res = await fetch('/api/reset_episode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId })
  });
  const data = await res.json();
  lastState = data.state;
  drawState(data.state);
  updateStats(data.stats);
  updateStatus('Running', 'success');
}

// ── Update Stats Panel ──
function updateStats(stats) {
  document.getElementById('statEpisodes').textContent = stats.total_episodes;
  document.getElementById('statSuccess').textContent = stats.success_rate + '%';
  document.getElementById('statCollision').textContent = stats.collision_rate + '%';

  const historyDiv = document.getElementById('episodeHistory');
  historyDiv.innerHTML = stats.history.slice().reverse().map(h => `
    <div class="episode-item ${h.result}">
      <span>Episode ${h.episode}</span>
      <span>${h.result.toUpperCase()} (${h.steps} steps)</span>
    </div>
  `).join('') || '<small class="text-muted">No episodes completed yet</small>';
}

// ── Canvas Drawing ──
function drawState(state) {
  const W = canvas.width, H = canvas.height;
  const scale = W / state.world_size;

  // Background
  ctx.fillStyle = '#1a2332';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= state.world_size; i++) {
    ctx.beginPath();
    ctx.moveTo(i * scale, 0); ctx.lineTo(i * scale, H);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * scale); ctx.lineTo(W, i * scale);
    ctx.stroke();
  }

  // Trajectory trail
  if (state.trajectory && state.trajectory.length > 1) {
    ctx.strokeStyle = 'rgba(46, 117, 182, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.trajectory.forEach((pt, i) => {
      const x = pt[0] * scale, y = pt[1] * scale;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Static obstacles
  state.static_obstacles.forEach(obs => {
    ctx.fillStyle = '#888';
    ctx.beginPath();
    ctx.arc(obs.x * scale, obs.y * scale, obs.r * scale, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // Dynamic obstacles
  state.dynamic_obstacles.forEach(obs => {
    ctx.fillStyle = '#FF4757';
    ctx.beginPath();
    ctx.arc(obs.x * scale, obs.y * scale, obs.r * scale, 0, 2 * Math.PI);
    ctx.fill();
    // Direction indicator
    ctx.strokeStyle = '#FF4757';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(obs.cx * scale, obs.cy * scale, obs.radius * scale, 0, 2 * Math.PI);
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // Goal
  const gx = state.goal.x * scale, gy = state.goal.y * scale;
  ctx.fillStyle = '#FFD700';
  ctx.beginPath();
  ctx.arc(gx, gy, 14, 0, 2 * Math.PI);
  ctx.fill();
  ctx.strokeStyle = '#FFF';
  ctx.lineWidth = 2;
  ctx.stroke();
  // Pulsing ring
  ctx.strokeStyle = 'rgba(255, 215, 0, 0.4)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(gx, gy, 24, 0, 2 * Math.PI);
  ctx.stroke();

  // Robot
  const rx = state.robot.x * scale, ry = state.robot.y * scale;
  const theta = state.robot.theta;

  // LIDAR rays (subtle)
  ctx.strokeStyle = 'rgba(0, 217, 255, 0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 24; i++) {
    const angle = theta + (2 * Math.PI * i / 24);
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(rx + Math.cos(angle) * 60, ry + Math.sin(angle) * 60);
    ctx.stroke();
  }

  // Robot body
  ctx.fillStyle = '#00D9FF';
  ctx.beginPath();
  ctx.arc(rx, ry, 12, 0, 2 * Math.PI);
  ctx.fill();
  ctx.strokeStyle = '#FFF';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Direction indicator
  ctx.strokeStyle = '#FFF';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(rx, ry);
  ctx.lineTo(rx + Math.cos(theta) * 20, ry + Math.sin(theta) * 20);
  ctx.stroke();

  // World boundary
  ctx.strokeStyle = '#2E75B6';
  ctx.lineWidth = 3;
  ctx.strokeRect(2, 2, W - 4, H - 4);
}

// ── Algo Switch Mid-Simulation ──
document.getElementById('algoSelect').addEventListener('change', async (e) => {
  if (!sessionId) return;
  const newAlgo = e.target.value;
  const res = await fetch('/api/switch_algo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, algo: newAlgo })
  });
  const data = await res.json();
  lastState = data.state;
  drawState(data.state);
  updateStats(data.stats);
  document.getElementById('infoBanner').innerHTML =
    `🔄 Switched to <strong>${newAlgo.toUpperCase()}</strong> agent. Stats are tracked cumulatively for comparison.`;
});
