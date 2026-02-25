const logBox = document.getElementById("log-box");
const logPath = document.getElementById("log-path");
const syncStatus = document.getElementById("sync-status");
const logFollow = document.getElementById("log-follow");
const processList = document.getElementById("process-list");
const processCount = document.getElementById("process-count");
const heartbeatEl = document.getElementById("heartbeat");
const stateBox = document.getElementById("state-box");

let followLogs = true;
function atBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 10;
}

logBox.addEventListener("scroll", () => {
  if (!logFollow) {
    return;
  }
  followLogs = atBottom(logBox);
  logFollow.checked = followLogs;
});

async function refreshLogs() {
  const res = await axios.get("/api/logs");
  logPath.textContent = res.data.path || "";
  const wasAtBottom = atBottom(logBox);
  logBox.textContent = (res.data.lines || []).join("\n");
  if (logFollow.checked && followLogs && wasAtBottom) {
    logBox.scrollTop = logBox.scrollHeight;
  }
}

async function refreshState() {
  const res = await axios.get("/api/state");
  stateBox.textContent = JSON.stringify(res.data, null, 2);
  const state = res.data.state || {};
  document.getElementById("st-status").textContent = state.status || "-";
  document.getElementById("st-today").textContent = state.today || "-";
  document.getElementById("st-target").textContent = state.target || "-";
  document.getElementById("st-pid").textContent = res.data.pid ?? "-";
  document.getElementById("st-run-started").textContent = state.run_started_at || "-";
  document.getElementById("st-run-finished").textContent = state.run_finished_at || "-";
  document.getElementById("st-rc").textContent = state.last_return_code ?? "-";
  const hb = (res.data && res.data.heartbeat) || {};
  document.getElementById("st-heartbeat").textContent = hb.age_seconds == null ? "-" : `${hb.age_seconds}s`;
  if (!res.data.running) {
    heartbeatEl.textContent = "heartbeat: daemon stopped";
    heartbeatEl.className = "small text-danger mt-1";
  } else if (hb.stale) {
    heartbeatEl.textContent = `heartbeat: stale (${hb.age_seconds ?? "-"}s)`;
    heartbeatEl.className = "small text-warning mt-1";
  } else {
    heartbeatEl.textContent = `heartbeat: ok (${hb.age_seconds ?? 0}s)`;
    heartbeatEl.className = "small text-success mt-1";
  }
}

async function refreshProcesses() {
  const res = await axios.get("/api/processes");
  const items = res.data.items || [];
  processCount.textContent = `Running: ${items.length}`;
  if (!items.length) {
    processList.innerHTML = "<div class='text-muted'>No scheduler daemon process</div>";
    return;
  }
  processList.innerHTML = items
    .map((item) => {
      return `
        <div class="process-card">
          <div class="pid">PID: ${item.pid}</div>
          <div class="text-muted">start: ${item.start || "-"}</div>
          <div class="text-muted">cmd: ${(item.cmd || "").replace(/</g, "&lt;")}</div>
        </div>
      `;
    })
    .join("");
}

async function callAction(url) {
  const res = await axios.post(url);
  return res.data;
}

async function loadConfig() {
  const res = await axios.get("/api/config");
  const cfg = res.data.config || {};
  const meta = res.data.meta || { read_only: [], types: {} };
  const readonly = new Set(meta.read_only || []);
  const keys = Object.keys(cfg).sort();
  const list = document.getElementById("config-list");

  const groups = { bool: [], data: [], readonly: [] };
  keys.forEach((key) => {
    const val = cfg[key];
    const type = meta.types && meta.types[key] ? meta.types[key] : "str";
    const ro = readonly.has(key);
    if (ro) {
      groups.readonly.push({ key, val, type, ro });
    } else if (type === "bool") {
      groups.bool.push({ key, val, type, ro });
    } else {
      groups.data.push({ key, val, type, ro });
    }
  });

  function renderItem({ key, val, type, ro }) {
    let input = "";
    if (type === "bool") {
      input = `<input type="checkbox" class="form-check-input" data-key="${key}" data-type="bool" ${val ? "checked" : ""} ${ro ? "disabled" : ""}>`;
    } else {
      const inputType = type === "int" || type === "float" ? "number" : "text";
      input = `<input type="${inputType}" class="form-control form-control-sm" data-key="${key}" data-type="${type}" value="${val}" ${ro ? "disabled" : ""}>`;
    }
    return `
      <div class="config-item ${ro ? "readonly" : ""}">
        <div class="label">${key}</div>
        ${input}
      </div>
    `;
  }

  function renderSection(items) {
    if (!items.length) return "";
    return `
      <div class="config-section">
        <div class="config-grid">
          ${items.map(renderItem).join("")}
        </div>
      </div>
    `;
  }

  list.innerHTML = renderSection(groups.bool) + renderSection(groups.data) + renderSection(groups.readonly);
}

async function saveConfig() {
  const list = document.getElementById("config-list");
  const inputs = list.querySelectorAll("input[data-key]");
  const payload = {};
  inputs.forEach((el) => {
    const key = el.getAttribute("data-key");
    const type = el.getAttribute("data-type");
    if (el.disabled) return;
    if (type === "bool") {
      payload[key] = el.checked;
    } else if (type === "int") {
      payload[key] = el.value === "" ? null : parseInt(el.value, 10);
    } else if (type === "float") {
      payload[key] = el.value === "" ? null : parseFloat(el.value);
    } else {
      payload[key] = el.value;
    }
  });
  await axios.post("/api/config", payload);
  syncStatus.textContent = "Config saved";
}

document.getElementById("btn-start").addEventListener("click", async () => {
  await callAction("/api/start");
  syncStatus.textContent = "Scheduler started";
  await tick();
});

document.getElementById("btn-restart").addEventListener("click", async () => {
  await callAction("/api/restart");
  syncStatus.textContent = "Scheduler restarted";
  await tick();
});

document.getElementById("btn-stop").addEventListener("click", async () => {
  await callAction("/api/stop");
  syncStatus.textContent = "Scheduler stopped";
  await tick();
});

document.getElementById("btn-shutdown").addEventListener("click", async () => {
  await callAction("/api/shutdown");
  setTimeout(() => {
    window.close();
  }, 300);
});

document.getElementById("btn-save-config").addEventListener("click", async () => {
  await saveConfig();
  await loadConfig();
});

async function tick() {
  await refreshLogs();
  await refreshProcesses();
  await refreshState();
}

loadConfig();
tick();
setInterval(tick, 3000);
