const logBox = document.getElementById("log-box");
const logPath = document.getElementById("log-path");
const holdingsEl = document.getElementById("holdings");
const syncStatus = document.getElementById("sync-status");
const logFollow = document.getElementById("log-follow");
const logDaySelect = document.getElementById("log-day");
const processList = document.getElementById("process-list");
const processCount = document.getElementById("process-count");

let followLogs = true;
function atBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 10;
}

logBox.addEventListener("scroll", () => {
  if (!logFollow) {
    return;
  }
  followLogs = atBottom(logBox);
  if (logFollow) {
    logFollow.checked = followLogs;
  }
});

async function refreshLogs() {
  const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
  const res = await axios.get("/api/logs", { params: selectedDay ? { day: selectedDay } : {} });
  logPath.textContent = res.data.path || "";
  const wasAtBottom = atBottom(logBox);
  logBox.textContent = (res.data.lines || []).join("\n");
  if (logFollow ? logFollow.checked : true) {
    if (followLogs && wasAtBottom) {
      logBox.scrollTop = logBox.scrollHeight;
    }
  }
}

async function loadLogDays() {
  if (!logDaySelect) return;
  const prev = logDaySelect.value;
  const res = await axios.get("/api/log_days");
  const days = res.data.days || [];
  const currentDay = res.data.current_day || "";
  logDaySelect.innerHTML = days
    .map((d) => `<option value="${d}">${d}</option>`)
    .join("");
  if (days.includes(prev)) {
    logDaySelect.value = prev;
  } else if (days.includes(currentDay)) {
    logDaySelect.value = currentDay;
  } else if (days.length) {
    logDaySelect.value = days[0];
  }
}

async function refreshHoldings() {
  const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
  const res = await axios.get("/api/holdings", { params: selectedDay ? { day: selectedDay } : {} });
  const rows = res.data.rows || [];
  if (!rows.length) {
    holdingsEl.innerHTML = "<div class='text-muted'>No holdings</div>";
    return;
  }
  holdingsEl.innerHTML = rows
    .map((row) => {
      const w = row.weight == null ? "" : Number(row.weight).toFixed(6);
      return `
        <div class="holding-card">
          <div class="sym">${row.symbol}</div>
          <div class="text-end">
            <div>weight: ${w}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

async function refreshProcesses() {
  const res = await axios.get("/api/processes");
  const items = res.data.items || [];
  if (processCount) {
    processCount.textContent = `Running: ${items.length}`;
  }
  if (!processList) return;
  if (!items.length) {
    processList.innerHTML = "<div class='text-muted'>No live processes</div>";
    return;
  }
  processList.innerHTML = items
    .map((item) => {
      const cmd = (item.cmd || "").replace(/.*liveLaunch\\\\/, "liveLaunch\\");
      return `
        <div class="process-card">
          <div class="pid">PID: ${item.pid}</div>
          <div class="text-muted">start: ${item.start || "-"}</div>
          <div class="text-muted">cmd: ${cmd}</div>
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
  const hiddenRootKeys = new Set(["start", "target"]);
  const list = document.getElementById("config-list");

  const groups = { bool: [], data: [], readonly: [] };

  function inferType(val) {
    if (typeof val === "boolean") return "bool";
    if (typeof val === "number") return Number.isInteger(val) ? "int" : "float";
    if (Array.isArray(val)) return "json";
    if (val !== null && typeof val === "object") return "object";
    return "str";
  }

  function pushItem(path, val, ro) {
    const type = inferType(val);
    if (type === "object") {
      const childKeys = Object.keys(val || {}).sort();
      childKeys.forEach((k) => pushItem(`${path}.${k}`, val[k], ro));
      return;
    }
    const item = { key: path, val, type, ro };
    if (ro) {
      groups.readonly.push(item);
    } else if (type === "bool") {
      groups.bool.push(item);
    } else {
      groups.data.push(item);
    }
  }

  const keys = Object.keys(cfg).sort();
  keys.forEach((key) => {
    if (hiddenRootKeys.has(key)) {
      return;
    }
    const ro = readonly.has(key);
    pushItem(key, cfg[key], ro);
  });

  function renderItem({ key, val, type, ro }) {
    let input = "";
    if (type === "bool") {
      input = `<input type="checkbox" class="form-check-input" data-key="${key}" data-type="bool" ${val ? "checked" : ""} ${ro ? "disabled" : ""}>`;
    } else {
      if (type === "json") {
        const jsonVal = JSON.stringify(val);
        input = `<input type="text" class="form-control form-control-sm" data-key="${key}" data-type="json" value='${jsonVal}' ${ro ? "disabled" : ""}>`;
      } else {
        const inputType = type === "int" || type === "float" ? "number" : "text";
        const safeVal = val == null ? "" : String(val);
        input = `<input type="${inputType}" class="form-control form-control-sm" data-key="${key}" data-type="${type}" value="${safeVal}" ${ro ? "disabled" : ""}>`;
      }
    }
    return `
        <div class="config-item ${ro ? "readonly" : ""}">
          <div class="label">${key}</div>
          ${input}
        </div>
      `;
  }

  function renderSection(title, items) {
    if (!items.length) return "";
    return `
      <div class="config-section">
        <div class="config-grid">
          ${items.map(renderItem).join("")}
        </div>
      </div>
    `;
  }

  list.innerHTML =
    renderSection("", groups.bool) +
    renderSection("", groups.data) +
    renderSection("", groups.readonly);
}

async function saveConfig() {
  const list = document.getElementById("config-list");
  const inputs = list.querySelectorAll("input[data-key]");
  const payload = {};

  function setByPath(obj, dottedKey, value) {
    const parts = dottedKey.split(".");
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i += 1) {
      const p = parts[i];
      if (!cur[p] || typeof cur[p] !== "object" || Array.isArray(cur[p])) {
        cur[p] = {};
      }
      cur = cur[p];
    }
    cur[parts[parts.length - 1]] = value;
  }

  inputs.forEach((el) => {
    const key = el.getAttribute("data-key");
    const type = el.getAttribute("data-type");
    if (el.disabled) return;
    let value;
    if (type === "bool") {
      value = el.checked;
    } else if (type === "int") {
      value = el.value === "" ? null : parseInt(el.value, 10);
    } else if (type === "float") {
      value = el.value === "" ? null : parseFloat(el.value);
    } else if (type === "json") {
      try {
        value = el.value === "" ? [] : JSON.parse(el.value);
      } catch (err) {
        value = [];
      }
    } else {
      value = el.value;
    }
    setByPath(payload, key, value);
  });
  await axios.post("/api/config", payload);
  syncStatus.textContent = "Config saved";
}

document.getElementById("btn-open").addEventListener("click", async () => {
  await callAction("/api/start_open");
  syncStatus.textContent = "Start Open triggered";
  refreshLogs();
});

document.getElementById("btn-restart").addEventListener("click", async () => {
  await callAction("/api/restart_scheduler");
  syncStatus.textContent = "Scheduler restarted";
  refreshLogs();
});

document.getElementById("btn-stop").addEventListener("click", async () => {
  await callAction("/api/emergency_stop");
  syncStatus.textContent = "Emergency Stop triggered";
  refreshLogs();
});

document.getElementById("btn-sync").addEventListener("click", async () => {
  const res = await callAction("/api/sync_holdings");
  if (res.ok) {
    syncStatus.textContent = `Synced holdings: ${res.count}`;
  } else {
    syncStatus.textContent = `Sync failed: ${res.error || ""}`;
  }
  refreshHoldings();
  refreshLogs();
});

document.getElementById("btn-shutdown").addEventListener("click", async () => {
  await callAction("/api/shutdown");
  setTimeout(() => {
    window.close();
  }, 300);
});

document.getElementById("btn-save-config").addEventListener("click", async () => {
  await saveConfig();
});

if (logDaySelect) {
  logDaySelect.addEventListener("change", async () => {
    await refreshLogs();
    await refreshHoldings();
  });
}

async function tick() {
  await refreshLogs();
  await refreshHoldings();
}

loadLogDays().then(() => tick());
loadConfig();
setInterval(tick, 3000);
setInterval(loadLogDays, 30000);
refreshProcesses();
setInterval(refreshProcesses, 10000);
