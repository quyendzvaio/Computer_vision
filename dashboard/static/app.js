/**
 * CV Safety Monitor — Dashboard JS
 * Handles WebSocket connection, camera grid rendering, alert bar updates.
 */
(function () {
  'use strict';

  // --- State ---
  const cameras = {};       // camera_id -> { canvas, ctx, card }
  let ws = null;
  let reconnectTimer = null;

  // --- DOM ---
  const grid = document.getElementById('camera-grid');
  const alertBar = document.getElementById('alert-bar');
  const statusDot = document.getElementById('status-dot');
  const latestAlertEl = document.getElementById('latest-alert-text');

  // --- WebSocket ---
  function connectWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/dashboard`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('[WS] Connected');
      statusDot.classList.remove('alarm');
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      fetchCameras();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
      } catch (e) {
        console.warn('[WS] Invalid message:', event.data);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected — reconnecting in 3s');
      statusDot.classList.add('alarm');
      reconnectTimer = setTimeout(connectWS, 3000);
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
    };
  }

  // --- Message Handler ---
  function handleMessage(msg) {
    switch (msg.type) {
      case 'violation':
        showAlert(msg.violation);
        break;
      case 'preview':
        updatePreview(msg.camera_id, msg.frame_base64);
        break;
      default:
        console.log('[WS] Unknown message type:', msg.type);
    }
  }

  // --- Alert Bar ---
  function showAlert(violation) {
    // Flash the alert bar
    alertBar.classList.remove('flash');
    void alertBar.offsetWidth; // reflow
    alertBar.classList.add('flash');

    // Update latest alert text
    const sevClass = violation.severity === 'HIGH' ? 'sev-high' : 'sev-medium';
    latestAlertEl.innerHTML = `<span class="${sevClass}">[${violation.type}]</span> Camera ${violation.camera_id} — ${new Date(violation.timestamp).toLocaleTimeString()}`;

    // Flash the matching camera card
    const card = document.querySelector(`[data-camera="${violation.camera_id}"]`);
    if (card) {
      const tag = card.querySelector('.violation-tag');
      if (tag) {
        tag.textContent = violation.type;
        tag.classList.add('show');
        setTimeout(() => tag.classList.remove('show'), 3000);
      }
    }

    // Auto-dismiss flash after animation
    setTimeout(() => alertBar.classList.remove('flash'), 2000);
  }

  // --- Preview Frame Update ---
  function updatePreview(cameraId, base64Frame) {
    let cam = cameras[cameraId];
    if (!cam) {
      cam = createCameraCard(cameraId);
      cameras[cameraId] = cam;
    }

    const img = new Image();
    img.onload = () => {
      cam.canvas.width = img.width;
      cam.canvas.height = img.height;
      cam.ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${base64Frame}`;
  }

  // --- Camera Grid ---
  function createCameraCard(cameraId) {
    const card = document.createElement('div');
    card.className = 'camera-card';
    card.setAttribute('data-camera', cameraId);

    const header = document.createElement('div');
    header.className = 'header';
    header.innerHTML = `
      <span class="cam-id">&#x1F4F7; ${cameraId}</span>
      <span class="status live">LIVE</span>
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 416;
    canvas.height = 416;

    const tag = document.createElement('div');
    tag.className = 'violation-tag';

    card.appendChild(header);
    card.appendChild(canvas);
    card.appendChild(tag);
    grid.appendChild(card);

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#444';
    ctx.font = '14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for stream...', canvas.width / 2, canvas.height / 2);

    return { canvas, ctx, card };
  }

  // --- Fetch initial camera list ---
  async function fetchCameras() {
    try {
      const resp = await fetch('/api/cameras');
      const list = await resp.json();
      list.forEach(cam => {
        if (!cameras[cam.id]) {
          cameras[cam.id] = createCameraCard(cam.id);
        }
      });
    } catch (e) {
      console.warn('Failed to fetch camera list:', e);
    }
  }

  // --- Startup ---
  connectWS();
})();
