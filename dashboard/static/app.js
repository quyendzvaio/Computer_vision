/**
 * CV Safety Monitor — Dashboard Client
 * WebSocket realtime updates, camera grid, alert system, stats.
 */
(function () {
  'use strict';

  // --- State ---
  const cameras = new Map();    // camera_id -> { canvas, ctx, card, badge, placeholder }
  const violationCount = { today: 0 };
  let ws = null;
  let reconnectTimer = null;
  const uptimeStart = Date.now();
  let uptimeTimer = null;

  // --- DOM Cache ---
  const $ = (sel) => document.querySelector(sel);
  const grid = $('#camera-grid');
  const alertBanner = $('#alert-banner');
  const mainContent = $('#main-content');
  const wsIndicator = $('#ws-indicator');
  const statCameras = $('#stat-cameras');
  const statViolations = $('#stat-violations');
  const statActiveCameras = $('#stat-active-cameras');
  const statTotalViolations = $('#stat-total-violations');
  const statUptime = $('#stat-uptime');
  const statRoiCount = $('#stat-roi-count');

  // --- Uptime Timer ---
  function updateUptime() {
    const elapsed = Math.floor((Date.now() - uptimeStart) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    statUptime.textContent = h + ':' + m;
  }
  uptimeTimer = setInterval(updateUptime, 10000);
  updateUptime();

  // --- WebSocket ---
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws/dashboard');

    ws.onopen = function () {
      console.log('[WS] Connected');
      wsIndicator.className = 'dot-indicator live';
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      fetchCameras();
      fetchViolationCount();
    };

    ws.onmessage = function (event) {
      try {
        handleMessage(JSON.parse(event.data));
      } catch (e) {
        console.warn('[WS] Bad message:', e);
      }
    };

    ws.onclose = function () {
      console.log('[WS] Disconnected — reconnecting in 2s');
      wsIndicator.className = 'dot-indicator warning';
      reconnectTimer = setTimeout(connectWS, 2000);
    };

    ws.onerror = function () {
      // onclose fires after onerror, reconnect is handled there
    };
  }

  // --- Message Router ---
  function handleMessage(msg) {
    switch (msg.type) {
      case 'violation':
        onViolation(msg.violation);
        break;
      case 'preview':
        onPreview(msg.camera_id, msg.frame_base64);
        break;
    }
  }

  // --- Violation Handler ---
  function onViolation(v) {
    // Update counts
    violationCount.today++;
    updateViolationStats();

    // Show alert banner
    alertBanner.classList.add('active');
    $('#alert-type').textContent = v.type.replace(/_/g, ' ');
    $('#alert-camera').textContent = 'Camera: ' + v.camera_id;
    $('#alert-time').textContent = new Date(v.timestamp).toLocaleTimeString();
    mainContent.classList.add('alert-active');

    // Auto-dismiss banner after 4s
    clearTimeout(alertBanner._timeout);
    alertBanner._timeout = setTimeout(function () {
      alertBanner.classList.remove('active');
      mainContent.classList.remove('alert-active');
    }, 4000);

    // Flash the matching camera card
    var cam = cameras.get(v.camera_id);
    if (cam) {
      cam.card.classList.add('violation-flash');
      cam.badge.textContent = v.type;
      cam.badge.classList.add('visible');
      setTimeout(function () {
        cam.card.classList.remove('violation-flash');
      }, 1200);
      setTimeout(function () {
        cam.badge.classList.remove('visible');
      }, 4000);
    }
  }

  // --- Preview Handler ---
  function onPreview(cameraId, b64) {
    var cam = cameras.get(cameraId);
    if (!cam) {
      cam = createCameraCard(cameraId);
      cameras.set(cameraId, cam);
      updateStats();
    }

    var img = new Image();
    img.onload = function () {
      cam.canvas.width = cam.canvas.naturalWidth || img.width;
      cam.canvas.height = cam.canvas.naturalHeight || img.height;
      cam.ctx.drawImage(img, 0, 0);
      // Hide placeholder on first frame
      if (cam.placeholder) {
        cam.placeholder.style.display = 'none';
      }
    };
    img.src = 'data:image/jpeg;base64,' + b64;
  }

  // --- Camera Card Factory ---
  function createCameraCard(cameraId) {
    var card = document.createElement('div');
    card.className = 'camera-card';
    card.setAttribute('data-camera', cameraId);

    card.innerHTML =
      '<div class="card-header">' +
        '<div class="cam-name">' +
          '<span class="cam-icon">&#x1F4F7;</span>' +
          '<span>' + cameraId + '</span>' +
        '</div>' +
        '<div class="cam-status live">' +
          '<span class="pulse-dot"></span>' +
          '<span>Live</span>' +
        '</div>' +
      '</div>' +
      '<div class="canvas-wrap">' +
        '<canvas></canvas>' +
        '<div class="placeholder">' +
          '<span class="cam-icon-lg">&#x1F4F7;</span>' +
          '<span>Waiting for stream…</span>' +
        '</div>' +
      '</div>' +
      '<div class="violation-badge"></div>';

    grid.appendChild(card);

    var canvas = card.querySelector('canvas');
    var ctx = canvas.getContext('2d');

    // Initial dark background
    canvas.width = 416;
    canvas.height = 312;
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    var badge = card.querySelector('.violation-badge');
    var placeholder = card.querySelector('.placeholder');

    return { canvas: canvas, ctx: ctx, card: card, badge: badge, placeholder: placeholder };
  }

  // --- Stats ---
  function updateStats() {
    statActiveCameras.textContent = cameras.size;
    statCameras.textContent = cameras.size;
  }

  function updateViolationStats() {
    statTotalViolations.textContent = violationCount.today;
    statViolations.textContent = violationCount.today;
  }

  // --- API Calls ---
  async function fetchCameras() {
    try {
      var resp = await fetch('/api/cameras');
      var list = await resp.json();
      list.forEach(function (cam) {
        if (!cameras.has(cam.id)) {
          cameras.set(cam.id, createCameraCard(cam.id));
        }
      });
      updateStats();
    } catch (e) {
      console.warn('Failed to load cameras:', e);
    }
  }

  async function fetchViolationCount() {
    try {
      var resp = await fetch('/api/violations?limit=1000');
      var data = await resp.json();
      var rows = data.violations || [];
      violationCount.today = rows.length;
      updateViolationStats();
    } catch (e) {
      console.warn('Failed to load violation count:', e);
    }
  }

  async function fetchRoiCount() {
    try {
      var resp = await fetch('/api/cameras');
      var camList = await resp.json();
      var count = 0;
      for (var i = 0; i < camList.length; i++) {
        try {
          var r = await fetch('/api/roi/' + camList[i].id);
          if (r.ok) count++;
        } catch (e) {
          // ROI not found — skip
        }
      }
      statRoiCount.textContent = count;
    } catch (e) {
      console.warn('Failed to load ROI count:', e);
    }
  }

  // --- Init ---
  connectWS();
  fetchRoiCount();
})();
