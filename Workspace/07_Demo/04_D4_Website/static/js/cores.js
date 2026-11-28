// Mục "Năm lõi được so sánh": 5 thẻ, mỗi thẻ một sơ đồ bên trong lõi.
// Bấm tên lõi ở Hình 1.3 hay bấm thẻ => mở đúng sơ đồ; liên kết #core-<lõi> mở thẳng thẻ đó.
const CORES = ["lstm", "gru", "transformer", "s4d", "mamba"];

function show(core, scroll) {
  if (!CORES.includes(core)) return;
  for (const k of CORES) {
    const on = k === core;
    const tab = document.getElementById(`ctab-${k}`);
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    document.getElementById(`core-${k}`).hidden = !on;
  }
  if (scroll) document.getElementById("cores").scrollIntoView({behavior: "smooth", block: "start"});
}

document.addEventListener("click", (e) => {
  const link = e.target.closest("[data-core]");
  if (!link) return;
  e.preventDefault();
  const fromTabs = Boolean(link.closest(".core-tabs"));
  // không đổi location.hash: app.js giữ lưu vực đang chọn ở #basin=<ID>
  show(link.dataset.core, !fromTabs);
});

// mũi tên trái, phải, Home, End chuyển thẻ (mẫu tablist của WAI-ARIA)
document.querySelector(".core-tabs")?.addEventListener("keydown", (e) => {
  const i = CORES.indexOf(e.target.dataset.core);
  if (i < 0) return;
  const next = {ArrowRight: (i + 1) % CORES.length, ArrowLeft: (i + CORES.length - 1) % CORES.length,
                Home: 0, End: CORES.length - 1}[e.key];
  if (next === undefined) return;
  e.preventDefault();
  show(CORES[next], false);
  document.getElementById(`ctab-${CORES[next]}`).focus();
});

const hash = location.hash.match(/^#core-(\w+)$/);
if (hash) show(hash[1], true);
