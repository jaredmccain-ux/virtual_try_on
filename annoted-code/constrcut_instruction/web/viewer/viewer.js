(function () {
  const samples = window.SHARE_SAMPLES || [];
  const state = {
    filtered: samples.slice(),
    activeId: null,
    jsonTab: "source",
    langTab: "en",
  };

  const els = {
    setupWarning: document.getElementById("setup-warning"),
    stats: document.getElementById("stats"),
    filterMode: document.getElementById("filter-mode"),
    filterScene: document.getElementById("filter-scene"),
    filterWearingType: document.getElementById("filter-wearing-type"),
    filterEdit: document.getElementById("filter-edit"),
    filterSearch: document.getElementById("filter-search"),
    listCount: document.getElementById("list-count"),
    sampleList: document.getElementById("sample-list"),
    content: document.getElementById("content"),
  };

  function showSetupWarning(message) {
    if (!els.setupWarning) return;
    els.setupWarning.classList.remove("hidden");
    els.setupWarning.textContent = message;
  }

  function detectSetupIssue() {
    if (!samples.length) {
      showSetupWarning(
        "未加载样本数据。请先将 constrcut_instruction_share.zip 完整解压到文件夹，再打开解压目录中的 index.html；不要从压缩包内直接双击打开。"
      );
      return;
    }
    const href = window.location.href || "";
    if (/\.zip/i.test(href) || /Compressed/i.test(href)) {
      showSetupWarning(
        "检测到你可能从压缩包内直接打开了页面。请先完整解压 zip，确保 index.html 与 images/ 文件夹在同一目录，再从解压后的 index.html 打开。"
      );
    }
  }

  function escapeHtml(text) {
    return String(text ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function countBy(samplesSubset, key) {
    const counts = {};
    for (const sample of samplesSubset) {
      const value = sample[key] || "—";
      counts[value] = (counts[value] || 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => a[0].localeCompare(b[0]));
  }

  function initFilters() {
    const editTypes = [...new Set(samples.map((s) => s.edit_type_id).filter(Boolean))].sort();
    for (const id of editTypes) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      els.filterEdit.appendChild(opt);
    }

    const scenes = [...new Set(samples.map((s) => s.scene_id).filter(Boolean))].sort();
    for (const scene of scenes) {
      const opt = document.createElement("option");
      opt.value = scene;
      opt.textContent = scene;
      els.filterScene.appendChild(opt);
    }

    const paired = samples.filter((s) => s.pair_mode === "paired").length;
    const unpair = samples.filter((s) => s.pair_mode === "unpair").length;
    const scenePills = countBy(samples, "scene_id")
      .map(([scene, count]) => `<span class="scene-pill">${escapeHtml(scene)}<strong>${count}</strong></span>`)
      .join("");

    els.stats.innerHTML = [
      `<div class="stat-chip">总样本<strong>${samples.length}</strong></div>`,
      `<div class="stat-chip">paired<strong>${paired}</strong></div>`,
      `<div class="stat-chip">unpair<strong>${unpair}</strong></div>`,
      `<div class="stat-chip" style="width:100%;">场景分布<div class="scene-stats">${scenePills}</div></div>`,
    ].join("");
  }

  function applyFilters() {
    const mode = els.filterMode.value;
    const scene = els.filterScene.value;
    const wearingType = els.filterWearingType.value;
    const edit = els.filterEdit.value;
    const q = els.filterSearch.value.trim().toLowerCase();

    state.filtered = samples.filter((sample) => {
      if (mode !== "all" && sample.pair_mode !== mode) return false;
      if (scene !== "all" && sample.scene_id !== scene) return false;
      if (wearingType !== "all" && sample.wearing_type !== wearingType) return false;
      if (edit !== "all" && sample.edit_type_id !== edit) return false;
      if (q) {
        const haystack = [
          sample.sample_id,
          sample.person_anchor_id,
          sample.garment_donor_id,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });

    if (state.activeId && !state.filtered.some((s) => s.sample_id === state.activeId)) {
      state.activeId = null;
    }
    renderList();
    renderDetail();
  }

  function renderList() {
    els.listCount.textContent = String(state.filtered.length);
    els.sampleList.innerHTML = state.filtered
      .map((sample) => {
        const active = sample.sample_id === state.activeId ? " active" : "";
        return `
          <button class="sample-item${active}" data-id="${escapeHtml(sample.sample_id)}">
            <div class="id">${escapeHtml(sample.sample_id)}</div>
            <div class="meta">
              <span class="tag ${escapeHtml(sample.pair_mode)}">${escapeHtml(sample.pair_mode)}</span>
              <span class="tag scene">${escapeHtml(sample.scene_id || "-")}</span>
              <span class="tag">${escapeHtml(sample.edit_type_id || "-")}</span>
            </div>
          </button>`;
      })
      .join("");

    els.sampleList.querySelectorAll(".sample-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeId = btn.dataset.id;
        renderList();
        renderDetail();
      });
    });

    const activeBtn = els.sampleList.querySelector(".sample-item.active");
    if (activeBtn) {
      activeBtn.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  function renderChanges(sample) {
    if (!sample.changes || !sample.changes.length) {
      return `<div class="instruction">无编辑项</div>`;
    }
    return `
      <div class="changes">
        ${sample.changes
          .map(
            (c) => `
          <div class="change-row">
            <div class="dim">${escapeHtml(c.dimension)}</div>
            <div class="attr">${escapeHtml(c.attribute)}</div>
            <div class="from">${escapeHtml(c.from)}</div>
            <div class="to">${escapeHtml(c.to)}</div>
          </div>`
          )
          .join("")}
      </div>`;
  }

  function parseInstructionSegments(text) {
    if (!text) return [];
    const segments = [];
    const pattern = /(\[EDIT[^\]]*\]|\[PRESERVE[^\]]*\])/g;
    let lastIndex = 0;
    let match;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        const base = text.slice(lastIndex, match.index).trim();
        if (base) {
          segments.push({ kind: "base", label: "BASE", text: base });
        }
      }
      const marker = match[1];
      const start = match.index + marker.length;
      const next = text.slice(start).search(/\[EDIT|\[PRESERVE/);
      const body = (next >= 0 ? text.slice(start, start + next) : text.slice(start)).trim();
      if (marker.startsWith("[EDIT")) {
        segments.push({ kind: "edit", label: marker, text: body });
      } else {
        segments.push({ kind: "preserve", label: marker, text: body });
      }
      lastIndex = next >= 0 ? start + next : text.length;
    }

    const tail = text.slice(lastIndex).trim();
    if (tail) {
      segments.push({ kind: "base", label: "BASE", text: tail });
    }

    if (!segments.length && text.trim()) {
      segments.push({ kind: "base", label: "BASE", text: text.trim() });
    }
    return segments;
  }

  function renderInstruction(text) {
    const segments = parseInstructionSegments(text);
    if (!segments.length) {
      return `<div class="instruction">—</div>`;
    }
    return `
      <div class="instruction-block">
        ${segments
          .map(
            (segment) => `
          <div class="instruction-segment ${escapeHtml(segment.kind)}">
            <div class="label">${escapeHtml(segment.label)}</div>
            ${escapeHtml(segment.text)}
          </div>`
          )
          .join("")}
      </div>`;
  }

  function renderMetaChips(sample) {
    const chips = [
      ["mode", sample.pair_mode],
      ["scene", sample.scene_id || "—"],
      ["edit", `${sample.edit_type_id || "—"} · ${sample.edit_type_label || ""}`.trim()],
      ["region", sample.active_region || "—"],
      ["state", sample.person_upper_state || "—"],
      ["class", sample.garment_class || "—"],
    ];
    if (sample.pair_mode === "unpair") {
      chips.push(["anchor", sample.person_anchor_id || "—"]);
      chips.push(["donor", sample.garment_donor_id || "—"]);
    }
    return chips
      .map(
        ([label, value]) =>
          `<span class="meta-chip">${escapeHtml(label)}<strong>${escapeHtml(value)}</strong></span>`
      )
      .join("");
  }

  function renderInstructionPanel(sample) {
    const text = state.langTab === "zh" ? sample.instruction_zh : sample.instruction_en;
    return `
      <div class="panel compact">
        <h2 class="section-title">编辑指令 instruction</h2>
        <div class="instruction-tabs">
          <button class="lang-btn${state.langTab === "en" ? " active" : ""}" data-lang="en">English</button>
          <button class="lang-btn${state.langTab === "zh" ? " active" : ""}" data-lang="zh">中文</button>
        </div>
        ${renderInstruction(text)}
      </div>`;
  }

  function renderJsonPanel(sample) {
    const tabs = [
      ["source", "source_attributes"],
      ["target", "target_attributes"],
      ["preserve", "preserved_by_dimension"],
      ["spec", "instruction_spec"],
    ];
    const active = tabs.find(([key]) => key === state.jsonTab) || tabs[0];
    const payload = sample[active[1]];

    return `
      <div class="panel json-panel">
        <h2 class="section-title">结构化 JSON</h2>
        <div class="tabs">
          ${tabs
            .map(
              ([key, label]) =>
                `<button class="tab-btn${state.jsonTab === key ? " active" : ""}" data-tab="${key}">${label}</button>`
            )
            .join("")}
        </div>
        <pre class="json-block">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
      </div>`;
  }

  function renderUnpairMeta(sample) {
    return "";
  }

  function renderDetail() {
    const sample = state.filtered.find((s) => s.sample_id === state.activeId);
    if (!sample) {
      els.content.innerHTML = `<div class="empty">请从左侧选择一条样本</div>`;
      return;
    }

    els.content.innerHTML = `
      <div class="detail-view">
        <div class="detail-header">
          <h2 class="title">${escapeHtml(sample.sample_id)}</h2>
          <div class="meta-row">${renderMetaChips(sample)}</div>
        </div>

        <div class="detail-main">
          <div class="detail-images">
            <div class="image-card">
              <img src="${escapeHtml(sample.person_image)}" alt="person" loading="lazy" />
              <div class="image-label">person</div>
            </div>
            <div class="image-card">
              <img src="${escapeHtml(sample.garment_image)}" alt="garment" loading="lazy" />
              <div class="image-label">garment</div>
            </div>
          </div>

          <div class="detail-body">
            <div class="detail-split">
              <div class="panel compact">
                <h2 class="section-title">编辑属性 changes</h2>
                ${renderChanges(sample)}
              </div>
              ${renderInstructionPanel(sample)}
            </div>
            ${renderJsonPanel(sample)}
          </div>
        </div>
      </div>
    `;

    els.content.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.jsonTab = btn.dataset.tab;
        renderDetail();
      });
    });

    els.content.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.langTab = btn.dataset.lang;
        renderDetail();
      });
    });
  }

  els.filterMode.addEventListener("change", applyFilters);
  els.filterScene.addEventListener("change", applyFilters);
  els.filterWearingType.addEventListener("change", applyFilters);
  els.filterEdit.addEventListener("change", applyFilters);
  els.filterSearch.addEventListener("input", applyFilters);

  detectSetupIssue();
  initFilters();
  applyFilters();
  if (state.filtered.length) {
    state.activeId = state.filtered[0].sample_id;
    renderList();
    renderDetail();
  }
})();
