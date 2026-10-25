/* Upscaler page client.
   Everything authoritative — the queue, each image's planned output size and
   the batch job — lives in the Django app. This script renders it, sends the
   settings, and polls a running batch. Inside the desktop app the native
   pieces (file dialogs, message boxes, drag & drop) come from the Qt host. */
(() => {
  "use strict";

  const config = JSON.parse(document.getElementById("upscaler-config").textContent);
  const $ = (id) => document.getElementById(id);
  const LOG_LINES = 2000;
  const POLL_MS = 350;

  const el = {
    queueList: $("queue-list"), queueEmpty: $("queue-empty"), queueCount: $("queue-count"),
    previewFrame: $("preview-frame"), previewTitle: $("preview-title"),
    previewDims: $("preview-dims"), previewNote: $("preview-note"),
    outputDir: $("output-dir"), saveBeside: $("save-beside"), folderHint: $("folder-hint"),
    device: $("device"), fp16: $("fp16"), tile: $("tile"),
    modeScale: $("mode-scale"), modeExact: $("mode-exact"), factor: $("scale-factor"),
    exactW: $("exact-width"), exactH: $("exact-height"), match: $("btn-match"),
    summary: $("size-summary"), conflict: $("conflict-box"), conflictText: $("conflict-text"),
    diagramPad: $("diagram-pad"), diagramCrop: $("diagram-crop"),
    detailPad: $("detail-pad"), detailCrop: $("detail-crop"), padColor: $("pad-color"),
    format: $("output-format"), quality: $("jpg-quality"), suffix: $("suffix"), rename: $("rename-batch"),
    start: $("btn-start"), cancel: $("btn-cancel"), status: $("status"),
    progress: $("progress"), progressChunk: $("progress-chunk"), progressText: $("progress-text"),
    log: $("log"), logCount: $("log-count"),
    addImages: $("btn-add-images"), addFolder: $("btn-add-folder"),
    remove: $("btn-remove"), clear: $("btn-clear"), browse: $("btn-browse"),
  };
  el.rows = document.createElement("div");
  el.rows.className = "rows";
  el.queueList.append(el.rows);

  const state = {
    queue: [],              // items from the server
    rows: new Map(),        // id -> planned output for the current settings
    rowEls: new Map(),      // id -> row element (kept, so thumbnails never reload)
    plan: null,             // last plan response
    selected: new Set(),
    anchor: null,
    running: false,
    starting: false,        // a Start request is in flight
    results: {},            // id -> per-image result of the current/last batch
    resultsKey: "",
    fitKey: "",             // frame + mismatched images the Pad/Crop choice was made for
    fitPending: false,
    logNext: 0,
    logLines: 0,
    autoSuffix: config.defaults.suffix,
    planTimer: 0,
    planSeq: 0,
    showFitError: false,
  };

  // -- server calls -----------------------------------------------------------
  function csrfToken() {
    const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    return m ? decodeURIComponent(m[1]) : "";
  }

  async function api(path, body) {
    const init = { credentials: "same-origin", headers: {} };
    if (body !== undefined) {
      init.method = "POST";
      init.headers["Content-Type"] = "application/json";
      init.headers["X-CSRFToken"] = csrfToken();
      init.body = JSON.stringify(body);
    }
    const res = await fetch(config.api + path, init);
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (!res.ok) {
      const err = new Error(data.error || `The Upscaler server answered ${res.status}.`);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  const thumbUrl = (id) => `${config.api}thumb/${encodeURIComponent(id)}/`;
  const previewUrl = (id) => `${config.api}preview/${encodeURIComponent(id)}/`;

  // -- small helpers ----------------------------------------------------------
  const dims = (wh) => `${wh[0]} × ${wh[1]}`;
  const roundHalfUp = (x) => Math.floor(x + 0.5);
  const formatFactor = (f) => String(Math.round(f * 100) / 100);
  const make = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function log(text) {
    const atBottom = el.log.scrollTop + el.log.clientHeight >= el.log.scrollHeight - 4;
    el.log.append(document.createTextNode(text + "\n"));
    while (el.log.childNodes.length > LOG_LINES) el.log.firstChild.remove();
    state.logLines += 1;
    el.logCount.textContent = `(${state.logLines})`;
    if (atBottom) el.log.scrollTop = el.log.scrollHeight;
  }

  async function showMessage(kind, title, text) {
    if (config.has_dialogs) {
      try { await api("dialog/message/", { kind, title, text }); return; } catch (_) { /* fall back */ }
    }
    const dlg = $("message-dialog");
    $("message-title").textContent = title;
    $("message-text").textContent = text;
    dlg.showModal();
    await new Promise((resolve) => dlg.addEventListener("close", resolve, { once: true }));
  }

  function askForPaths(title) {
    const dlg = $("path-dialog");
    const input = $("path-input");
    $("path-title").textContent = title;
    input.value = "";
    dlg.returnValue = "";
    dlg.showModal();
    return new Promise((resolve) => {
      dlg.addEventListener("close", () => {
        resolve(dlg.returnValue === "ok"
          ? input.value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean) : []);
      }, { once: true });
    });
  }

  function reportError(err) {
    log(`✗ ${err.message}`);
    showMessage("warning", (err.data && err.data.title) || "Upscaler", err.message);
  }

  // -- settings -----------------------------------------------------------------
  function sizeMode() { return el.modeExact.checked ? "exact" : "scale"; }

  function fitChoice() {
    const checked = document.querySelector('input[name="fit"]:checked');
    return checked ? checked.value : "";
  }

  function settings() {
    return {
      size_mode: sizeMode(),
      scale_factor: el.factor.value,
      exact_width: el.exactW.value,
      exact_height: el.exactH.value,
      fit: fitChoice(),
      pad_color: el.padColor.value,
      output_dir: el.outputDir.value,
      save_beside: el.saveBeside.checked,
      device: el.device.value,
      fp16: el.fp16.checked,
      tile: el.tile.value,
      output_format: el.format.value,
      jpg_quality: el.quality.value,
      suffix: el.suffix.value,
      rename_batch: el.rename.checked,
    };
  }

  function applyDefaults() {
    const d = config.defaults;
    el.outputDir.value = d.output_dir;
    el.outputDir.placeholder = config.default_output_dir;
    el.saveBeside.checked = d.save_beside;
    el.device.value = d.device;
    el.fp16.checked = d.fp16;
    el.tile.value = d.tile;
    (d.size_mode === "exact" ? el.modeExact : el.modeScale).checked = true;
    el.factor.value = d.scale_factor;
    el.factor.min = config.limits.min_factor;
    el.factor.max = config.limits.max_factor;
    el.exactW.max = el.exactH.max = config.limits.max_side;
    el.exactW.value = d.exact_width;
    el.exactH.value = d.exact_height;
    el.padColor.value = d.pad_color;
    el.format.value = d.output_format;
    el.quality.value = d.jpg_quality;
    el.suffix.value = d.suffix;
    el.rename.checked = d.rename_batch;
    el.folderHint.textContent =
      `Images from "Add Folder…" always save to <folder>/${config.output_dir_name}/, ` +
      "keeping each image's subfolder path.";
  }

  function autoSuffix() {
    if (sizeMode() === "exact") {
      return el.exactW.value && el.exactH.value ? `_${el.exactW.value}x${el.exactH.value}` : "_exact";
    }
    const f = parseFloat(el.factor.value);
    return Number.isFinite(f) ? `_x${formatFactor(f)}` : "_x4";
  }

  // The suffix follows the size settings until the user types their own.
  function syncSuffix() {
    const next = autoSuffix();
    if (el.suffix.value === state.autoSuffix) el.suffix.value = next;
    state.autoSuffix = next;
  }

  function updateControls() {
    const r = state.running;
    const exact = sizeMode() === "exact";
    for (const node of document.querySelectorAll(".settings input, .settings select, .settings button")) {
      node.disabled = r;
    }
    if (!r) {
      el.outputDir.disabled = el.saveBeside.checked;
      el.browse.disabled = el.saveBeside.checked;
      el.factor.disabled = exact;
      el.exactW.disabled = el.exactH.disabled = !exact;
      el.match.disabled = !exact || !referenceItem();
      el.quality.disabled = el.format.value !== "jpg";
      el.suffix.disabled = el.rename.checked;
      el.padColor.disabled = fitChoice() !== "pad";
    }
    for (const b of [el.addImages, el.addFolder]) b.disabled = r;
    el.clear.disabled = r || !state.queue.length;
    el.remove.disabled = r || state.selected.size === 0;
    el.start.disabled = r || state.starting;
    el.cancel.disabled = !r;
  }

  function onSizeChanged() {
    syncSuffix();
    updateControls();
    schedulePlan();
  }

  // -- queue rendering ----------------------------------------------------------
  function referenceItem() {
    for (const item of state.queue) if (state.selected.has(item.id)) return item;
    return state.queue[0] || null;
  }

  const STATUS = {
    processing: ["Processing…", "processing"], done: ["Saved", "done"],
    failed: ["Failed", "failed"], cancelled: ["Cancelled", "cancelled"],
  };

  function buildRow(item) {
    const row = make("div", "queue-row");
    row.dataset.id = item.id;
    row.setAttribute("role", "option");
    const img = make("img", "thumb");
    img.alt = "";
    img.loading = "lazy";
    img.decoding = "async";
    img.src = thumbUrl(item.id);
    img.addEventListener("error", () => img.classList.add("broken"));
    const src = make("div", "cell");
    src.append(make("div", "name", item.name), make("div", "src"));
    const out = make("div", "cell out");
    const outDims = make("div", "out-dims");
    outDims.append(make("span", "arrow", "→"), make("span", "v"));
    out.append(outDims, make("div", "note"));
    row.append(img, src, out, make("span", "pill"));
    return row;
  }

  function updateRow(rowEl, item) {
    const row = state.rows.get(item.id);
    rowEl.classList.toggle("selected", state.selected.has(item.id));
    rowEl.setAttribute("aria-selected", state.selected.has(item.id) ? "true" : "false");
    rowEl.classList.toggle("needs-choice", !!row && row.kind === "conflict");
    rowEl.title = item.folder_root
      ? `${item.path}\nFrom folder ${item.folder_root} — saves under its "${config.output_dir_name}" subfolder`
      : item.path;
    rowEl.querySelector(".src").textContent = `${dims([item.width, item.height])} · ${item.ratio} ${item.orientation}`;
    rowEl.querySelector(".out-dims .v").textContent = row && row.output ? dims(row.output) : "—";
    const note = rowEl.querySelector(".note");
    note.textContent = row ? row.note : "";
    if (row && row.warning) note.append(" ", make("span", "warning", row.warning));
    const pill = rowEl.querySelector(".pill");
    const res = state.results[item.id];
    const [text, cls] = (res && STATUS[res.status]) || ["", ""];
    pill.textContent = text;
    pill.className = `pill ${cls}`;
    pill.title = res ? (res.output || res.error || "") : "";
  }

  function renderQueue() {
    const ids = new Set(state.queue.map((i) => i.id));
    for (const [id, rowEl] of state.rowEls) {
      if (!ids.has(id)) { rowEl.remove(); state.rowEls.delete(id); }
    }
    state.queue.forEach((item, index) => {
      let rowEl = state.rowEls.get(item.id);
      if (!rowEl) { rowEl = buildRow(item); state.rowEls.set(item.id, rowEl); }
      if (el.rows.children[index] !== rowEl) el.rows.insertBefore(rowEl, el.rows.children[index] || null);
      updateRow(rowEl, item);
    });
    el.queueEmpty.hidden = state.queue.length > 0;
    const n = state.queue.length;
    el.queueCount.textContent = n ? `${n} image${n === 1 ? "" : "s"}${state.selected.size ? ` · ${state.selected.size} selected` : ""}` : "No images yet";
    renderPreview();
    updateControls();
  }

  function setQueue(queue) {
    state.queue = queue;
    const ids = new Set(queue.map((i) => i.id));
    for (const id of [...state.selected]) if (!ids.has(id)) state.selected.delete(id);
    if (state.anchor && !ids.has(state.anchor)) state.anchor = null;
    if (!state.selected.size && queue.length) { state.selected.add(queue[0].id); state.anchor = queue[0].id; }
    if (!state.running) el.status.textContent = `${queue.length} image(s) queued`;
    renderQueue();
    schedulePlan();
  }

  // -- result frames: the picture inside its output frame -------------------------
  function drawFrame(box, geometry, fit, padColor, imageUrl) {
    box.replaceChildren();
    if (!geometry) {
      box.append(make("div", "placeholder", "Add an image to see how it will be saved."));
      return;
    }
    const small = box.classList.contains("small");
    const W = box.clientWidth || (small ? 140 : 320);
    const H = box.clientHeight || (small ? 66 : 200);
    const margin = small ? 3 : 14;
    const [fw, fh] = geometry.frame, [cw, ch] = geometry.content, [ox, oy] = geometry.offset;
    const minX = Math.min(0, ox), minY = Math.min(0, oy);
    const maxX = Math.max(fw, ox + cw), maxY = Math.max(fh, oy + ch);
    const k = Math.min((W - 2 * margin) / (maxX - minX), (H - 2 * margin) / (maxY - minY));
    const tx = (W - (maxX - minX) * k) / 2 - minX * k;
    const ty = (H - (maxY - minY) * k) / 2 - minY * k;
    const place = (node, x, y, w, h) => {
      node.style.left = `${tx + x * k}px`;
      node.style.top = `${ty + y * k}px`;
      node.style.width = `${Math.max(1, w * k)}px`;
      node.style.height = `${Math.max(1, h * k)}px`;
      box.append(node);
      return node;
    };
    if (fit === "pad") place(make("div", `pad-${padColor}`), 0, 0, fw, fh);
    let pic;
    if (imageUrl) {
      // Reuse the element for the same image so redraws never flash or refetch.
      if (!box._img || box._img.dataset.src !== imageUrl) {
        box._img = make("img", "pic");
        box._img.alt = "";
        box._img.dataset.src = imageUrl;
        box._img.src = imageUrl;
      }
      pic = box._img;
    } else {
      pic = make("div", "pic");
    }
    place(pic, ox, oy, cw, ch);
    if (fit === "crop") {
      place(make("div", "outside"), minX, minY, maxX - minX, -minY);   // above the frame
      place(make("div", "outside"), minX, fh, maxX - minX, maxY - fh); // below
      place(make("div", "outside"), minX, 0, -minX, fh);               // left
      place(make("div", "outside"), fw, 0, maxX - fw, fh);             // right
    }
    place(make("div", "frame"), 0, 0, fw, fh);
  }

  function renderPreview() {
    const item = referenceItem();
    const row = item && state.rows.get(item.id);
    if (!item) {
      drawFrame(el.previewFrame, null);
      el.previewTitle.textContent = "No image selected";
      el.previewDims.textContent = "";
      el.previewNote.textContent = "";
      return;
    }
    el.previewTitle.textContent = item.name;
    el.previewTitle.title = item.path;
    if (!row || !row.output) {
      drawFrame(el.previewFrame, { frame: [item.width, item.height], content: [item.width, item.height], offset: [0, 0] },
        "", "", previewUrl(item.id));
      el.previewDims.textContent = `${dims([item.width, item.height])} (${item.ratio} ${item.orientation})`;
      el.previewNote.textContent = row ? row.note : "";
      return;
    }
    const fit = row.kind === "pad" || row.kind === "crop" ? row.kind : "";
    drawFrame(el.previewFrame, row.geometry, fit, el.padColor.value, previewUrl(item.id));
    const same = row.kind === "scale" || row.kind === "exact" ? ` · still ${item.ratio}` : "";
    el.previewDims.textContent = `${dims([item.width, item.height])} → ${dims(row.output)}${same}`;
    el.previewNote.textContent = row.warning ? `${row.note}. ${row.warning}.` : row.note;
  }

  function renderConflict() {
    const plan = state.plan;
    const conflicts = plan && sizeMode() === "exact" ? plan.conflicts : 0;
    el.conflict.hidden = !conflicts;
    el.summary.textContent = conflicts || !plan ? "" : plan.summary;
    if (!conflicts) return;
    el.conflictText.textContent = plan.summary;
    const ref = state.queue.find((i) => state.selected.has(i.id) && state.rows.get(i.id)?.alternatives)
      || state.queue.find((i) => state.rows.get(i.id)?.alternatives);
    const alts = ref && state.rows.get(ref.id).alternatives;
    if (!alts) return;
    drawFrame(el.diagramPad, alts.pad.geometry, "pad", el.padColor.value, thumbUrl(ref.id));
    drawFrame(el.diagramCrop, alts.crop.geometry, "crop", el.padColor.value, thumbUrl(ref.id));
    el.detailPad.textContent = `${ref.name}: ${alts.pad.note}`;
    el.detailCrop.textContent = `${ref.name}: ${alts.crop.note}`;
  }

  function renderErrors(errors) {
    for (const node of document.querySelectorAll("[data-error-for]")) {
      const field = node.dataset.errorFor;
      const show = field !== "fit" || state.showFitError;
      node.textContent = show && errors[field] ? errors[field].join(" ") : "";
    }
    const fields = {
      scale_factor: el.factor, exact_width: el.exactW, exact_height: el.exactH,
      suffix: el.suffix, output_dir: el.outputDir, tile: el.tile, jpg_quality: el.quality,
    };
    for (const [field, input] of Object.entries(fields)) input.classList.toggle("invalid", !!errors[field]);
  }

  // -- planning -----------------------------------------------------------------
  function schedulePlan() {
    clearTimeout(state.planTimer);
    state.planTimer = setTimeout(refreshPlan, 120);
  }

  async function refreshPlan() {
    const seq = ++state.planSeq;
    let plan;
    try {
      plan = await api("plan/", { settings: settings() });
    } catch (err) {
      log(`✗ Could not preview the output sizes: ${err.message}`);
      return;
    }
    if (seq !== state.planSeq) return;      // a newer request is on its way
    state.plan = plan;
    state.rows = new Map(plan.items.map((row) => [row.id, row]));
    // A Pad/Crop choice is made for one frame and one set of mismatched images.
    // If either changes, ask again instead of silently reusing the old answer.
    const key = `${el.exactW.value}x${el.exactH.value}|` +
      plan.items.filter((row) => row.alternatives).map((row) => row.id).join(",");
    if (state.fitPending) {
      state.fitKey = key;
      state.fitPending = false;
    } else if (sizeMode() === "exact" && fitChoice() && key !== state.fitKey) {
      for (const radio of document.querySelectorAll('input[name="fit"]')) radio.checked = false;
      log("The frame size or the images changed — choose Pad or Crop again.");
      updateControls();
      schedulePlan();
    }
    if (!plan.errors.fit) state.showFitError = false;
    renderErrors(plan.errors);
    renderQueue();
    renderConflict();
  }

  // -- adding / removing ------------------------------------------------------------
  async function handleAdded(data) {
    if (data.queue) setQueue(data.queue);
    const added = data.added;
    if (!added) return;
    for (const folder of added.folders) {
      log(`Folder mode: ${folder.count} image(s) from ${folder.path} will be saved under ` +
          `${folder.path}\\${added.output_dir_name} with subfolders preserved.`);
    }
    for (const bad of added.unreadable) log(`✗ Can't read ${bad.path}: ${bad.error}`);
    for (const path of added.missing) log(`✗ Not found: ${path}`);
    if (added.duplicates) log(`${added.duplicates} image(s) were already in the queue.`);
    if (added.empty_folders.length) {
      await showMessage("info", "No images found",
        `No supported images were found in:\n${added.empty_folders.join("\n")}\n\n` +
        `Nothing was added to the queue. (A previously generated '${added.output_dir_name}' ` +
        "output folder is skipped on purpose.)");
    } else if (added.added === 0) {
      log("No new supported images found in that selection.");
    }
  }

  async function addPaths(paths) {
    if (state.running || !paths || !paths.length) return;
    try { await handleAdded(await api("queue/add/", { paths })); } catch (err) { reportError(err); }
  }

  async function addImages() {
    try {
      if (config.has_dialogs) await handleAdded(await api("dialog/images/", {}));
      else await addPaths(await askForPaths("Add images"));
    } catch (err) { reportError(err); }
  }

  async function addFolder() {
    try {
      if (config.has_dialogs) await handleAdded(await api("dialog/folder/", {}));
      else await addPaths(await askForPaths("Add a folder"));
    } catch (err) { reportError(err); }
  }

  async function removeSelected() {
    if (!state.selected.size) return;
    try {
      const ids = [...state.selected];
      state.selected.clear();
      setQueue((await api("queue/remove/", { ids })).queue);
    } catch (err) { reportError(err); }
  }

  async function clearQueue() {
    try {
      state.results = {};
      state.resultsKey = "";
      setQueue((await api("queue/clear/", {})).queue);
    } catch (err) { reportError(err); }
  }

  async function browseOutput() {
    if (!config.has_dialogs) { el.outputDir.focus(); return; }
    try {
      const data = await api("dialog/output/", { current: el.outputDir.value });
      if (data.path) { el.outputDir.value = data.path; schedulePlan(); }
    } catch (err) { reportError(err); }
  }

  function matchSelectedShape() {
    const item = referenceItem();
    if (!item) return;
    let width = parseInt(el.exactW.value, 10);
    if (!Number.isFinite(width) || width < 1) width = item.width * 4;
    el.exactW.value = width;
    el.exactH.value = Math.max(1, roundHalfUp(width * item.height / item.width));
    onSizeChanged();
  }

  // -- selection --------------------------------------------------------------------
  function selectRow(id, event) {
    const order = state.queue.map((i) => i.id);
    if (event.shiftKey && state.anchor) {
      const [a, b] = [order.indexOf(state.anchor), order.indexOf(id)].sort((x, y) => x - y);
      if (!event.ctrlKey) state.selected.clear();
      for (const rid of order.slice(a, b + 1)) state.selected.add(rid);
    } else if (event.ctrlKey || event.metaKey) {
      if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id);
      state.anchor = id;
    } else {
      state.selected = new Set([id]);
      state.anchor = id;
    }
    renderQueue();
    renderConflict();
  }

  function moveSelection(step) {
    if (!state.queue.length) return;
    const order = state.queue.map((i) => i.id);
    const current = state.anchor ? order.indexOf(state.anchor) : -1;
    const next = Math.min(order.length - 1, Math.max(0, current + step));
    state.selected = new Set([order[next]]);
    state.anchor = order[next];
    renderQueue();
    renderConflict();
    state.rowEls.get(order[next])?.scrollIntoView({ block: "nearest" });
  }

  // -- batch --------------------------------------------------------------------------
  function setProgress(done, total) {
    const pct = total ? Math.round((done / total) * 100) : 0;
    el.progressChunk.style.width = `${pct}%`;
    el.progressText.textContent = total ? `${done} / ${total} · ${pct}%` : "";
    el.progress.setAttribute("aria-valuenow", String(pct));
  }

  function setRunning(running) {
    state.running = running;
    updateControls();
  }

  async function start() {
    if (!state.queue.length) {
      await showMessage("info", "Nothing to do", "Add some images first.");
      return;
    }
    if (state.starting) return;
    state.starting = true;
    updateControls();
    let data;
    try {
      data = await api("jobs/", { settings: settings() });
    } catch (err) {
      state.starting = false;
      updateControls();
      if (err.data && err.data.errors) {
        state.showFitError = true;
        renderErrors(err.data.errors);
        if (err.data.errors.fit) {
          el.conflict.classList.remove("flash");
          void el.conflict.offsetWidth;            // restart the animation
          el.conflict.classList.add("flash");
          el.conflict.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
      }
      await showMessage("warning", (err.data && err.data.title) || "Can't start", err.message);
      return;
    }
    state.results = {};
    state.resultsKey = "";
    state.logNext = 0;
    state.starting = false;
    setRunning(true);
    setProgress(0, state.queue.length);
    log(`Starting batch of ${state.queue.length} image(s)…`);
    applyJob(data.job);
  }

  async function cancel() {
    el.cancel.disabled = true;
    el.status.textContent = "Cancelling…";
    try { await api("jobs/current/cancel/", {}); } catch (err) { reportError(err); }
  }

  function jobStatus(job) {
    if (job.state === "cancelling") return "Cancelling…";
    switch (job.phase) {
      case "downloading": {
        const [done, total] = job.download || [0, 0];
        return total ? `Downloading model ${Math.floor((done * 100) / total)}%` : "Downloading model…";
      }
      case "starting":
      case "loading": return "Loading model…";
      case "processing": return `Processing ${job.current}…`;
      default: return `${job.done}/${job.total} done`;
    }
  }

  function applyJob(job) {
    for (const line of job.log) log(line);
    state.logNext = job.log_next;
    setProgress(job.done, job.total);
    // Rebuild rows only when a result changed, not on every poll.
    const resultsKey = JSON.stringify(job.results);
    if (resultsKey !== state.resultsKey) {
      state.resultsKey = resultsKey;
      state.results = job.results;
      renderQueue();
    }
    if (job.running) {
      el.status.textContent = jobStatus(job);
      setTimeout(poll, POLL_MS);
    } else {
      finish(job);
    }
  }

  async function poll() {
    try {
      const data = await api(`jobs/current/?since=${state.logNext}`);
      if (data.job) applyJob(data.job);
      else setRunning(false);
    } catch (err) {
      log(`(lost contact with the batch: ${err.message} — retrying)`);
      setTimeout(poll, POLL_MS * 4);
    }
  }

  function finish(job) {
    setRunning(false);
    const cancelled = job.state === "cancelled";
    el.status.textContent = `${cancelled ? "Cancelled" : "Finished"} — ${job.ok} ok, ${job.failed} failed`;
    if (job.state === "failed") {
      showMessage("warning", "Couldn't start", "The model could not be prepared. See the activity log for details.");
      $("log-panel").open = true;
    } else if (job.failed === 0) {
      showMessage("info", cancelled ? "Cancelled" : "Done",
        cancelled ? `Upscaled ${job.ok} image(s) before cancelling.` : `Upscaled ${job.ok} image(s).`);
    } else {
      $("log-panel").open = true;
      showMessage("warning", "Finished with errors", `${job.ok} succeeded, ${job.failed} failed. See the activity log.`);
    }
  }

  // -- wiring ---------------------------------------------------------------------------
  function bind() {
    el.addImages.addEventListener("click", addImages);
    el.addFolder.addEventListener("click", addFolder);
    el.remove.addEventListener("click", removeSelected);
    el.clear.addEventListener("click", clearQueue);
    el.browse.addEventListener("click", browseOutput);
    el.start.addEventListener("click", start);
    el.cancel.addEventListener("click", cancel);
    el.match.addEventListener("click", matchSelectedShape);

    // Select on press, like a native list (and robust to rows updating mid-click).
    el.rows.addEventListener("mousedown", (e) => {
      const row = e.button === 0 && e.target.closest(".queue-row");
      if (row) selectRow(row.dataset.id, e);
    });
    el.queueList.addEventListener("keydown", (e) => {
      if (e.key === "Delete" && !state.running) { e.preventDefault(); removeSelected(); }
      else if (e.key === "ArrowDown") { e.preventDefault(); moveSelection(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); moveSelection(-1); }
      else if (e.key === "a" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        state.selected = new Set(state.queue.map((i) => i.id));
        renderQueue();
      }
    });

    for (const radio of [el.modeScale, el.modeExact]) {
      radio.addEventListener("change", () => {
        if (sizeMode() === "exact" && !el.exactW.value && !el.exactH.value) {
          const item = referenceItem();      // start from a frame that fits the selected image
          if (item) { el.exactW.value = item.width * 4; el.exactH.value = item.height * 4; }
        }
        onSizeChanged();
      });
    }
    for (const input of [el.factor, el.exactW, el.exactH]) input.addEventListener("input", onSizeChanged);
    for (const radio of document.querySelectorAll('input[name="fit"]')) {
      radio.addEventListener("change", () => {
        state.fitPending = true;        // the next plan records what this choice covers
        updateControls();
        schedulePlan();
      });
    }
    el.padColor.addEventListener("change", () => { renderPreview(); renderConflict(); schedulePlan(); });
    el.saveBeside.addEventListener("change", updateControls);
    el.rename.addEventListener("change", updateControls);
    el.format.addEventListener("change", () => { updateControls(); schedulePlan(); });
    for (const input of [el.outputDir, el.suffix, el.tile, el.quality, el.device, el.fp16]) {
      input.addEventListener("input", schedulePlan);
    }

    // Keep the result frame sized to its box as the window changes.
    if (window.ResizeObserver) new ResizeObserver(() => renderPreview()).observe(el.previewFrame);

    // Explorer drops are handled natively by the desktop app (which knows the
    // real paths); a plain browser only gets file contents, so explain instead.
    document.addEventListener("dragover", (e) => e.preventDefault());
    document.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files.length) {
        showMessage("info", "Use Add Images…",
          "Dropped files can only be queued in the desktop app. Use Add Images… or Add Folder… instead.");
      }
    });
  }

  async function init() {
    applyDefaults();
    bind();
    updateControls();
    log(config.model_ready ? `Model ready: ${config.model_file}`
                           : "Model weights will download automatically on first run.");
    try {
      const data = await api(`state/?since=0`);
      setQueue(data.queue);
      if (!data.queue.length) el.status.textContent = "Idle";
      if (data.job && data.job.running) {   // page reloaded mid-batch: pick it back up
        setRunning(true);
        applyJob(data.job);
      }
    } catch (err) {
      reportError(err);
    }
  }

  // Called by the desktop host (drag & drop from Explorer).
  window.upscalerApp = { addPaths, refresh: refreshPlan };

  init();
})();
