/* 纯数据逻辑同时供浏览器和 Node 测试使用。 */
(function (root) {
  "use strict";
  const choices = ["主路", "副路", "字幕", "手改", "删除"];
  function valid(item, value) {
    return value && value.句号 === item.句号 && choices.includes(value.决定) &&
      typeof value.最终文本 === "string" && (value.决定 === "删除" || value.最终文本.trim()) &&
      ["清楚", "重叠", "听不清"].includes(value.听辨) && typeof value.备注 === "string" &&
      (!["主路", "副路", "字幕"].includes(value.决定) || (item[value.决定] && value.最终文本 === item[value.决定]));
  }
  function summary(snapshot) {
    return snapshot.map(({id, 文本, 起, 止}) => ({id, 文本, 起, 止}));
  }
  function restore(storage, key, snapshot, items) {
    try {
      const saved = JSON.parse(storage.getItem(key));
      if (!saved || JSON.stringify(summary(saved.snapshot)) !== JSON.stringify(summary(snapshot))) return {decisions: {}, available: true};
      const decisions = {};
      for (const item of items) {
        const value = saved.decisions?.[item.句号];
        if (value && value.句号 === item.句号 && [...choices, ""].includes(value.决定) &&
            typeof value.最终文本 === "string" && ["清楚", "重叠", "听不清"].includes(value.听辨) && typeof value.备注 === "string") {
          decisions[item.句号] = value;
        }
      }
      return {decisions, available: true};
    } catch (_) { return {decisions: {}, available: false}; }
  }
  function persist(storage, key, snapshot, decisions) {
    try { storage.setItem(key, JSON.stringify({snapshot, decisions})); return true; }
    catch (_) { return false; }
  }
  function exportResults(snapshot, items, decisions) {
    return {声音快照: snapshot, 声音摘要: summary(snapshot), 字幕参考: Object.fromEntries(items.map(item => [item.句号, item.字幕])), 结果: items.filter(item => valid(item, decisions[item.句号])).map(item => ({...decisions[item.句号]}))};
  }
  const api = {valid, restore, persist, exportResults};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.ListenSheet = api;
  if (typeof document === "undefined") return;

  const data = JSON.parse(document.getElementById("listen-data").textContent);
  const articles = [...document.querySelectorAll("article[data-index]")];
  const notice = document.getElementById("notice");
  let storage;
  try { storage = root.localStorage; } catch (_) { storage = null; }
  const restored = restore(storage, data.key, data.snapshot, data.items);
  const decisions = restored.decisions;
  let active = 0;
  let playing = null;
  const field = (card, name) => card.querySelector(`[data-field="${name}"]`);
  function storageWarning() { notice.textContent = "浏览器不能保存进度，关掉页面会丢失。离开前请导出回听结果。"; }
  if (!restored.available) storageWarning();
  function progress() {
    const total = data.items.filter(item => valid(item, decisions[item.句号])).length;
    document.getElementById("progress").textContent = `已处理 ${total} / ${data.items.length}`;
  }
  function decorate(index) {
    const card = articles[index], item = data.items[index], state = decisions[item.句号];
    const done = !!valid(item, state);
    card.classList.toggle("done", done);
    card.querySelector("[data-status]").textContent = done ? `已选：${state.决定}` : "待处理";
    for (const button of card.querySelectorAll("[data-choice]")) button.setAttribute("aria-pressed", String(done && button.dataset.choice === state.决定));
    field(card, "delete").checked = done && state.决定 === "删除";
    progress();
  }
  function save(index, choice) {
    const card = articles[index], item = data.items[index];
    if (["主路", "副路", "字幕"].includes(choice)) {
      if (!item[choice]) return;
      field(card, "text").value = item[choice];
    }
    const value = {句号: item.句号, 决定: choice, 最终文本: choice === "删除" ? "" : field(card, "text").value,
      听辨: field(card, "clarity").value, 备注: field(card, "note").value};
    decisions[item.句号] = value;
    if (!persist(storage, data.key, data.snapshot, decisions)) storageWarning();
    decorate(index);
  }
  function focusCard(index, moveFocus = false) {
    if (!articles.length) return;
    active = Math.max(0, Math.min(articles.length - 1, index));
    articles.forEach((card, i) => card.classList.toggle("active", i === active));
    if (moveFocus) { articles[active].querySelector("h2").focus(); articles[active].scrollIntoView({block: "start"}); }
  }
  function play(index, kind) {
    const card = articles[index];
    const target = card.querySelector(`[data-audio="${kind}"]`);
    if (playing === target && !target.paused) { target.pause(); return; }
    if (playing) playing.pause();
    playing = target;
    target.loop = field(card, "loop").checked;
    target.currentTime = 0;
    target.play().catch(() => { notice.textContent = "音频没能播放，请检查回听音频文件是否随页面保留。"; });
  }
  articles.forEach((card, index) => {
    const state = decisions[data.items[index].句号];
    if (state) {
      if (state.决定 !== "删除") field(card, "text").value = state.最终文本;
      field(card, "clarity").value = state.听辨;
      field(card, "note").value = state.备注;
    }
    decorate(index);
    card.addEventListener("focusin", () => focusCard(index));
    card.addEventListener("pointerdown", () => focusCard(index));
    card.addEventListener("click", event => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.dataset.choice) save(index, button.dataset.choice);
      if (button.dataset.play) play(index, button.dataset.play);
      if (button.hasAttribute("data-reset")) {
        delete decisions[data.items[index].句号];
        if (!persist(storage, data.key, data.snapshot, decisions)) storageWarning();
        decorate(index);
      }
    });
    field(card, "text").addEventListener("input", () => save(index, "手改"));
    field(card, "delete").addEventListener("change", event => save(index, event.target.checked ? "删除" : "手改"));
    for (const name of ["clarity", "note"]) field(card, name).addEventListener("input", () => {
      const old = decisions[data.items[index].句号];
      save(index, old ? old.决定 : "");
    });
    field(card, "loop").addEventListener("change", event => {
      for (const audio of card.querySelectorAll("audio")) audio.loop = event.target.checked;
    });
    for (const audio of card.querySelectorAll("audio")) audio.addEventListener("play", () => {
      if (playing && playing !== audio) playing.pause();
      playing = audio;
    });
  });
  progress();
  focusCard(0);
  document.addEventListener("keydown", event => {
    if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey || !articles.length) return;
    if (event.target.closest("input,textarea,select,[contenteditable=true]")) {
      if (event.key === "Escape") articles[active].querySelector("h2").focus();
      return;
    }
    const key = event.key.toLowerCase();
    if (![" ", "c", "1", "2", "3", "j", "k", "e"].includes(key)) return;
    // 空格在原生按钮、媒体控件上保留原生行为，避免触发两次。
    if (key === " " && event.target.closest("button,a,audio,summary")) return;
    event.preventDefault();
    if (key === " " || key === "c") play(active, key === "c" ? "context" : "sentence");
    if (["1", "2", "3"].includes(key)) save(active, ["主路", "副路", "字幕"][Number(key) - 1]);
    if (key === "j" || key === "k") focusCard(active + (key === "j" ? 1 : -1), true);
    if (key === "e") field(articles[active], "text").focus();
  });
  document.getElementById("export").addEventListener("click", () => {
    const value = exportResults(data.snapshot, data.items, decisions);
    const blob = new Blob([JSON.stringify(value, null, 2)], {type: "application/json;charset=utf-8"});
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = "回听结果.json";
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})(typeof window === "undefined" ? globalThis : window);
