"use strict";
/* 学术阅读器前端（零依赖，namespace 编译为经典脚本，file:// 可用）。
 *
 * 输入：app/reader.py 产出的 ReaderDocument（window.__READER_DOC__）。
 * 三件事：
 *   1. 正文 [n] → 参考文献条目；参考文献 → 每个引用处的回链。
 *   2. 标题树高亮：用"当前阅读行上/附近正文块的 rank"反推标题，而不是 offsetTop。
 *      双栏里右栏标题与左栏正文同高，offsetTop 与阅读顺序不一致；rank 才是阅读顺序。
 *   3. 双栏还原：每页按跨栏块切"栏带"，带内左右两列并排，保留每列自上而下的顺序。
 *      这是重排（reflow），不是原页像素级复刻——没有页面图像，也不该用绝对定位。
 *
 * 编译：npx --yes -p typescript@5.4.5 tsc reader.ts --target ES2020 --module none \
 *          --lib DOM,DOM.Iterable,ES2020 --strict
 */
var Reader;
(function (Reader) {
    const BAR_OFFSET = 76; // sticky 工具条高度 + 呼吸空间
    const READING_LINE = 96; // "当前阅读位置"距视口顶部的距离
    const STATUS_LABEL = {
        resolved: "已定位", partial: "部分编号缺失", missing: "参考文献表中不存在",
        ambiguous: "编号对应多条文献", invalid: "标记无法解析",
    };
    function esc(s) {
        return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }
    function reduceMotion() {
        return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    }
    /** 极简 Markdown 表格：只认管道表；不像表就原样交给 <pre>。 */
    function mdTable(md) {
        const lines = md.trim().split(/\r?\n/).filter((l) => l.trim().startsWith("|"));
        if (lines.length < 3)
            return `<pre class="table-raw">${esc(md)}</pre>`;
        const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
        const head = cells(lines[0]);
        const body = lines.slice(2).map(cells);
        const width = head.length;
        if (!width || body.some((r) => r.length !== width))
            return `<pre class="table-raw">${esc(md)}</pre>`;
        return `<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>` +
            `<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }
    class ReaderView {
        constructor(doc, mount) {
            this.mode = "flow";
            this.occById = new Map();
            this.refById = new Map();
            this.blockById = new Map();
            this.rankByAnchor = new Map();
            this.flatToc = [];
            this.cycle = new Map();
            this.metrics = [];
            this.refsTop = null;
            this.activeTocId = null;
            this.activeOccId = null;
            this.suppressUntil = 0;
            this.raf = 0;
            this.measureTimer = 0;
            this.toastTimer = 0;
            this.mounted = false;
            this.ro = null;
            // ───────────────────────────── 事件 ─────────────────────────────
            this.onClick = (e) => {
                const t = e.target;
                if (!t)
                    return;
                const occBtn = t.closest("[data-occ]");
                if (occBtn) {
                    e.preventDefault();
                    this.activateOccurrence(occBtn.dataset.occ ?? "");
                    return;
                }
                const back = t.closest("[data-backlink]");
                if (back) {
                    e.preventDefault();
                    this.backToOccurrence(back.dataset.backlink ?? "");
                    return;
                }
                const toc = t.closest("[data-toc]");
                if (toc) {
                    e.preventDefault();
                    this.onTocClick(toc);
                    return;
                }
                const modeBtn = t.closest(".mode-btn[data-mode]");
                if (modeBtn) {
                    e.preventDefault();
                    this.setMode(modeBtn.dataset.mode === "paged" ? "paged" : "flow");
                }
            };
            this.onKeydown = (e) => {
                if (e.key === "Escape")
                    this.hideToast();
            };
            this.onResize = () => { this.scheduleMeasure(); };
            this.cancelSuppress = () => { this.suppressUntil = 0; };
            this.scheduleMeasure = () => {
                window.clearTimeout(this.measureTimer);
                this.measureTimer = window.setTimeout(() => {
                    this.measure();
                    this.updateSpy(true);
                }, 80);
            };
            this.scheduleSpy = () => {
                if (this.raf)
                    return;
                this.raf = requestAnimationFrame(() => {
                    this.raf = 0;
                    this.updateSpy(false);
                });
            };
            this.doc = doc;
            this.mount = mount;
            for (const o of doc.citations?.occurrences ?? [])
                this.occById.set(o.id, o);
            for (const r of doc.references ?? [])
                this.refById.set(r.id, r);
            for (const b of doc.blocks ?? []) {
                this.blockById.set(b.id, b);
                this.rankByAnchor.set(b.anchor, b.rank);
            }
            const flat = (nodes) => {
                for (const n of nodes) {
                    this.flatToc.push(n);
                    flat(n.children ?? []);
                }
            };
            flat(doc.toc ?? []);
        }
        // ───────────────────────────── 生命周期 ─────────────────────────────
        mountView() {
            if (this.mounted)
                return;
            this.mounted = true;
            const title = this.doc.meta?.title ?? "未命名文献";
            this.mount.innerHTML = `
<div class="reader" data-mode="flow">
  <aside class="reader-toc" aria-label="文档目录">
    <div class="toc-head">
      <div class="toc-kicker">CONTENTS</div>
      <div class="toc-title" title="${esc(title)}">${esc(title)}</div>
    </div>
    <nav class="toc-scroll"><div class="toc-list"></div></nav>
    <div class="toc-stats"></div>
  </aside>
  <main class="reader-main">
    <div class="reader-bar">
      <div class="mode-switch" role="group" aria-label="版式切换">
        <button type="button" class="mode-btn" data-mode="flow" aria-pressed="true">流式阅读</button>
        <button type="button" class="mode-btn" data-mode="paged" aria-pressed="false">双栏还原</button>
      </div>
      <div class="bar-hint">点击正文 <b>[n]</b> 跳到参考文献；条目下的 <b>↩</b> 返回引用处</div>
    </div>
    <div class="warn-host"></div>
    <article class="reader-article" aria-label="论文正文"></article>
    <div class="refs-host"></div>
  </main>
  <div class="reader-toast" role="status" aria-live="polite" hidden></div>
</div>`;
            this.root = this.mount.querySelector(".reader");
            this.articleEl = this.mount.querySelector(".reader-article");
            this.tocListEl = this.mount.querySelector(".toc-list");
            this.tocScrollEl = this.mount.querySelector(".toc-scroll");
            this.refsEl = this.mount.querySelector(".refs-host");
            this.warnEl = this.mount.querySelector(".warn-host");
            this.toastEl = this.mount.querySelector(".reader-toast");
            this.renderToc();
            this.renderStats();
            this.renderWarnings();
            this.refsEl.innerHTML = this.renderRefs();
            this.renderArticle();
            this.root.addEventListener("click", this.onClick);
            document.addEventListener("keydown", this.onKeydown);
            window.addEventListener("scroll", this.scheduleSpy, { passive: true });
            window.addEventListener("resize", this.onResize);
            window.addEventListener("wheel", this.cancelSuppress, { passive: true });
            window.addEventListener("touchstart", this.cancelSuppress, { passive: true });
            window.addEventListener("load", this.scheduleMeasure);
            if (typeof ResizeObserver !== "undefined") {
                this.ro = new ResizeObserver(this.scheduleMeasure);
                this.ro.observe(this.articleEl);
                this.ro.observe(this.refsEl);
            }
            this.measure();
            if (location.hash)
                requestAnimationFrame(() => this.followHash());
            this.updateSpy(true);
        }
        destroy() {
            if (!this.mounted)
                return;
            this.root.removeEventListener("click", this.onClick);
            document.removeEventListener("keydown", this.onKeydown);
            window.removeEventListener("scroll", this.scheduleSpy);
            window.removeEventListener("resize", this.onResize);
            window.removeEventListener("wheel", this.cancelSuppress);
            window.removeEventListener("touchstart", this.cancelSuppress);
            window.removeEventListener("load", this.scheduleMeasure);
            if (this.ro) {
                this.ro.disconnect();
                this.ro = null;
            }
            window.clearTimeout(this.measureTimer);
            window.clearTimeout(this.toastTimer);
            if (this.raf)
                cancelAnimationFrame(this.raf);
            this.mounted = false;
        }
        // ───────────────────────────── 渲染：目录 ─────────────────────────────
        renderToc() {
            const walk = (nodes, depth) => {
                if (!nodes.length)
                    return "";
                return `<ul class="toc-ul" role="list">${nodes.map((n) => {
                    const pad = Math.min(depth, 5);
                    return `<li class="toc-li">
<button type="button" class="toc-link${pad === 0 ? " toc-root" : ""}" data-toc="${esc(n.anchor)}" data-node="${esc(n.id)}"
  style="--depth:${pad}" title="${esc(n.text)}${n.page ? ` · 第 ${n.page} 页` : ""}">
  <span class="toc-text">${esc(n.text || "（无标题）")}</span>
  <span class="toc-page">${n.page || ""}</span>
</button>${walk(n.children ?? [], depth + 1)}</li>`;
                }).join("")}</ul>`;
            };
            this.tocListEl.innerHTML = walk(this.doc.toc ?? [], 0) ||
                `<p class="toc-empty">未识别到标题树</p>`;
        }
        renderStats() {
            const s = this.doc.stats ?? {};
            const c = this.doc.citations?.stats;
            const bits = [`${s.blocks ?? 0} 个正文块`, `${s.references ?? 0} 条文献`];
            if (c) {
                const bad = c.missing + c.partial + c.ambiguous + c.invalid;
                bits.push(`引用 ${c.total} 处${bad ? `（${bad} 处需注意）` : ""}`);
            }
            const el = this.mount.querySelector(".toc-stats");
            if (el)
                el.innerHTML = bits.map((b) => `<span>${esc(b)}</span>`).join("");
        }
        renderWarnings() {
            const ws = this.doc.warnings ?? [];
            if (!ws.length) {
                this.warnEl.innerHTML = "";
                return;
            }
            this.warnEl.innerHTML = `<details class="warn-box"><summary>解析告警 ${ws.length} 条</summary>
<ul>${ws.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></details>`;
        }
        // ───────────────────────────── 渲染：正文块 ─────────────────────────────
        citeButton(s) {
            const occ = s.occurrence_id ? this.occById.get(s.occurrence_id) : undefined;
            const status = (s.status ?? occ?.status ?? "invalid");
            const targets = s.targets ?? occ?.targets ?? [];
            const label = `引用 ${s.text}，${STATUS_LABEL[status]}` + (targets.length ? `，跳转到 ${targets.length} 条参考文献` : "");
            const oid = s.occurrence_id ?? "";
            const idAttr = oid ? ` id="occ-${esc(oid)}"` : "";
            return `<button type="button" class="cite cite-${status}"${idAttr}
  data-occ="${esc(oid)}" aria-label="${esc(label)}" title="${esc(label)}">${esc(s.text)}</button>`;
        }
        segmentsHtml(b, text) {
            if (!b.segments?.length)
                return esc(text);
            return b.segments.map((s) => s.kind === "cite" ? this.citeButton(s) : esc(s.text)).join("");
        }
        renderBlock(b) {
            const id = ` id="${esc(b.anchor)}"`;
            const base = `class="block" data-rank="${b.rank}" data-block="${esc(b.id)}"`;
            switch (b.type) {
                case "title":
                    return `<h1${id} ${base} data-kind="title">${this.segmentsHtml(b, b.text)}</h1>`;
                case "heading": {
                    const lv = Math.min(6, Math.max(2, (b.level ?? 1) + 1));
                    const cls = `block block-heading level-${b.level ?? 1}`;
                    return `<h${lv}${id} class="${cls}" data-rank="${b.rank}" data-block="${esc(b.id)}">${this.segmentsHtml(b, b.text)}</h${lv}>`;
                }
                case "list_item":
                    return `<p${id} ${base}><span class="bullet" aria-hidden="true">•</span> ${this.segmentsHtml(b, b.text)}</p>`;
                case "formula": {
                    const body = b.latex || b.text;
                    const badge = b.latex_approx ? `<span class="formula-badge">LaTeX 近似</span>` : "";
                    return `<div${id} ${base} data-kind="formula">${badge}<code class="formula-code">${esc(body)}</code></div>`;
                }
                case "table": {
                    // 解析层的 table.text 有时是表头行而不是题注；有 caption 时以 caption 为准。
                    // caption === text 时仍走 segments，保留题注里的可点击引用（演示数据即如此）。
                    const capRaw = (b.caption && b.caption.trim()) ? b.caption : b.text;
                    const capHtml = capRaw === b.text ? this.segmentsHtml(b, b.text) : esc(capRaw);
                    const cap = capRaw ? `<figcaption class="block-caption">${capHtml}</figcaption>` : "";
                    const body = b.markdown ? mdTable(b.markdown) : `<pre class="table-raw">${esc(b.text)}</pre>`;
                    return `<figure${id} ${base} data-kind="table">${cap}${body}</figure>`;
                }
                case "figure": {
                    const img = b.image_path
                        ? `<img src="${esc(b.image_path)}" alt="${esc(b.caption || b.text || "插图")}" loading="lazy">`
                        : `<div class="figure-ph" role="img" aria-label="图片占位：${esc(b.caption || "无图注")}">图</div>`;
                    const capRaw = (b.caption || b.text || "").trim();
                    const capHtml = capRaw && capRaw === (b.text || "").trim()
                        ? this.segmentsHtml(b, b.text) : esc(capRaw);
                    const cap = capRaw ? `<figcaption class="block-caption">${capHtml}</figcaption>` : "";
                    return `<figure${id} ${base} data-kind="figure">${img}${cap}</figure>`;
                }
                case "figure_caption":
                case "table_caption":
                    return `<p${id} class="block block-caption" data-rank="${b.rank}" data-block="${esc(b.id)}">${this.segmentsHtml(b, b.text)}</p>`;
                case "unknown":
                    return `<p${id} class="block block-unknown" data-rank="${b.rank}" data-block="${esc(b.id)}">${this.segmentsHtml(b, b.text)}</p>`;
                default:
                    return `<p${id} ${base}>${this.segmentsHtml(b, b.text)}</p>`;
            }
        }
        renderArticle() {
            this.articleEl.innerHTML = this.mode === "paged" ? this.renderPaged() : this.renderFlow();
            this.restoreActiveMarker();
        }
        renderFlow() {
            return `<div class="flow">${(this.doc.blocks ?? []).map((b) => this.renderBlock(b)).join("")}</div>`;
        }
        /** 每页：非跨栏块按栏归栈，跨栏块切断栏带；带内左栏先于右栏。 */
        renderPaged() {
            const pages = (this.doc.pages ?? []).slice().sort((a, b) => a.page - b.page);
            const known = new Set();
            let html = "";
            for (const pg of pages) {
                known.add(pg.page);
                const blocks = (this.doc.blocks ?? []).filter((b) => b.page === pg.page).sort((a, b) => a.rank - b.rank);
                if (!blocks.length)
                    continue;
                html += `<section class="page-sheet" data-page="${pg.page}" aria-label="第 ${pg.page} 页">
<header class="page-meta"><span>第 ${pg.page} 页</span><span>${pg.columns > 1 ? `${pg.columns} 栏` : "单栏"} · ${Math.round(pg.width)}×${Math.round(pg.height)}pt${pg.estimated ? " · 估算" : ""}</span></header>
${this.pageBody(pg, blocks)}</section>`;
            }
            const orphan = (this.doc.blocks ?? []).filter((b) => !known.has(b.page));
            if (orphan.length) {
                html += `<section class="page-sheet" aria-label="补充内容">
<header class="page-meta"><span>补充</span><span>缺少页面几何</span></header>
<div class="flow">${orphan.map((b) => this.renderBlock(b)).join("")}</div></section>`;
            }
            return html;
        }
        pageBody(pg, blocks) {
            if (pg.columns <= 1)
                return `<div class="flow">${blocks.map((b) => this.renderBlock(b)).join("")}</div>`;
            const cols = Math.max(2, pg.columns);
            let out = "";
            let band = null;
            const flush = () => {
                if (band && band.some((c) => c.length)) {
                    out += `<div class="band" style="--cols:${cols}">` + band.map((col, i) => `<div class="col" data-col="${i}">${col.map((b) => this.renderBlock(b)).join("")}</div>`).join("") + `</div>`;
                }
                band = null;
            };
            for (const b of blocks) {
                if (b.span || b.column < 0) {
                    flush();
                    out += `<div class="band band-span">${this.renderBlock(b)}</div>`;
                }
                else {
                    if (!band)
                        band = Array.from({ length: cols }, () => []);
                    band[Math.min(cols - 1, Math.max(0, b.column))].push(b);
                }
            }
            flush();
            return out;
        }
        // ───────────────────────────── 渲染：参考文献 ─────────────────────────────
        safeHref(raw, prefix = "") {
            if (!raw)
                return null;
            try {
                const u = new URL(prefix ? prefix + raw : raw, location.href);
                return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
            }
            catch {
                return null;
            }
        }
        displayRefText(r) {
            const t = r.text ?? "";
            if (!r.number)
                return t;
            const m = t.match(/^\s*[\[［(（]\s*(\d{1,4})\s*[\]］)）]\s*/);
            if (m && String(parseInt(m[1], 10)) === r.number)
                return t.slice(m[0].length);
            return t;
        }
        snippet(blockId, start, end) {
            const text = this.blockById.get(blockId)?.text ?? "";
            const a = Math.max(0, start - 16), z = Math.min(text.length, end + 12);
            const core = text.slice(a, z).replace(/\s+/g, " ").trim();
            return `${a > 0 ? "…" : ""}${core}${z < text.length ? "…" : ""}`;
        }
        renderRefs() {
            const refs = this.doc.references ?? [];
            if (!refs.length)
                return "";
            const nums = Object.keys(this.doc.citations?.by_number ?? {}).length;
            const items = refs.map((r) => {
                const links = [];
                if (r.doi) {
                    const href = this.safeHref(r.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, ""), "https://doi.org/");
                    if (href)
                        links.push(`<a class="ref-ext" href="${esc(href)}" target="_blank" rel="noopener noreferrer">DOI</a>`);
                }
                if (r.arxiv) {
                    const href = this.safeHref(r.arxiv.replace(/^arxiv:\s*/i, ""), "https://arxiv.org/abs/");
                    if (href)
                        links.push(`<a class="ref-ext" href="${esc(href)}" target="_blank" rel="noopener noreferrer">arXiv</a>`);
                }
                // 只接受绝对 http(s)；裸域名若按相对路径解析会指到阅读器本地目录
                if (r.url && /^https?:\/\//i.test(r.url)) {
                    const href = this.safeHref(r.url);
                    if (href)
                        links.push(`<a class="ref-ext" href="${esc(href)}" target="_blank" rel="noopener noreferrer">链接</a>`);
                }
                const backs = (r.cite_ids ?? []).map((id) => {
                    const occ = this.occById.get(id);
                    if (!occ)
                        return "";
                    const page = this.blockById.get(occ.block_id)?.page ?? 0;
                    const snip = this.snippet(occ.block_id, occ.start, occ.end);
                    return `<button type="button" class="ref-back" data-backlink="${esc(id)}"
  title="${esc(snip)}" aria-label="返回第 ${page} 页的正文引用 ${esc(occ.raw)}"><span aria-hidden="true">↩</span> p.${page}<span class="ref-back-snip">${esc(snip)}</span></button>`;
                }).join("");
                return `<li id="${esc(r.id)}" class="ref-entry${r.ambiguous ? " is-ambiguous" : ""}" tabindex="-1">
  <span class="ref-num" aria-hidden="true">${esc(r.number ?? "—")}</span>
  <div class="ref-body">
    <p class="ref-text">${esc(this.displayRefText(r))}</p>
    <div class="ref-foot">
      <span class="ref-page">第 ${r.page} 页</span>
      ${r.ambiguous ? `<span class="ref-warn" title="该编号对应多条参考文献">编号重复</span>` : ""}
      ${links.join("")}
      <span class="ref-back-label">正文引用</span>
      <span class="ref-backlinks">${backs || `<span class="ref-none">未被正文引用</span>`}</span>
    </div>
  </div>
</li>`;
            }).join("");
            return `<section id="reader-references" class="reader-refs" aria-labelledby="reader-refs-title" tabindex="-1">
<div class="refs-head"><h2 id="reader-refs-title">参考文献</h2>
<span class="refs-count">${refs.length} 条 · ${nums} 个编号</span></div>
${(this.doc.citations?.unresolved_numbers ?? []).length
                ? `<p class="refs-alert">正文出现但参考文献表缺号：${esc(this.doc.citations.unresolved_numbers.join("、"))}</p>` : ""}
<ol class="ref-list">${items}</ol></section>`;
        }
        onTocClick(btn) {
            const anchor = btn.dataset.toc ?? "";
            const el = document.getElementById(anchor);
            if (!el)
                return;
            this.setActiveToc(btn.dataset.node ?? null, true);
            this.scrollToEl(el);
            try {
                history.replaceState(null, "", `#${anchor}`);
            }
            catch { /* file:// 某些浏览器限制 */ }
        }
        // ───────────────────────────── 引用跳转 ─────────────────────────────
        restoreActiveMarker() {
            if (!this.activeOccId)
                return;
            const el = document.getElementById(`occ-${this.activeOccId}`);
            if (el)
                el.classList.add("is-active");
        }
        activateOccurrence(id) {
            const occ = this.occById.get(id);
            if (!occ)
                return;
            this.markOccurrence(id);
            if (!occ.targets.length) {
                const why = occ.status === "invalid" ? "标记无法展开" :
                    `编号 ${occ.ref_ids.join("、") || occ.raw} 不在参考文献表中`;
                this.showToast(`<strong>${esc(occ.raw)}</strong>：${esc(why)}`, 5000);
                return;
            }
            const idx = this.cycle.get(id) ?? 0;
            this.cycle.set(id, idx + 1);
            const rid = occ.targets[idx % occ.targets.length];
            const ref = this.refById.get(rid);
            const el = document.getElementById(rid);
            if (!el)
                return;
            for (const n of this.root.querySelectorAll(".ref-entry.is-target"))
                n.classList.remove("is-target");
            el.classList.add("is-target");
            this.flash(el, "is-flash");
            this.scrollToEl(el);
            try {
                history.replaceState(null, "", `#${rid}`);
            }
            catch { /* ignore */ }
            if (el instanceof HTMLElement)
                el.focus({ preventScroll: true });
            const many = occ.targets.length > 1 || occ.status === "ambiguous";
            const pos = many ? `（${(idx % occ.targets.length) + 1}/${occ.targets.length}，该标记对应多条）` : "";
            const missing = occ.unresolved_ids.length ? `；编号 ${occ.unresolved_ids.join("、")} 缺失` : "";
            const label = ref?.number ? `[${ref.number}]` : "该条目";
            this.showToast(`已跳转到参考文献 ${esc(label)}${esc(pos + missing)}
<button type="button" class="toast-action" data-backlink="${esc(occ.id)}">↩ 返回引用处</button>`, 7000);
        }
        backToOccurrence(id) {
            const occ = this.occById.get(id);
            if (!occ)
                return;
            for (const n of this.root.querySelectorAll(".ref-entry.is-target"))
                n.classList.remove("is-target");
            this.markOccurrence(id);
            const marker = document.getElementById(`occ-${id}`);
            if (marker) {
                this.flash(marker, "is-flash");
                marker.focus({ preventScroll: true });
                this.scrollToEl(marker);
            }
            const page = this.blockById.get(occ.block_id)?.page;
            // 同时把 hash 移到出现处：否则参考文献条的 :target 会一直亮着
            try {
                history.replaceState(null, "", `#occ-${id}`);
            }
            catch { /* ignore */ }
            this.showToast(`已返回正文引用 <strong>${esc(occ.raw)}</strong>${page ? `（第 ${page} 页）` : ""}`, 4000);
        }
        markOccurrence(id) {
            this.activeOccId = id;
            for (const n of this.root.querySelectorAll(".cite.is-active"))
                n.classList.remove("is-active");
            document.getElementById(`occ-${id}`)?.classList.add("is-active");
        }
        scrollToEl(el) {
            const reduce = reduceMotion();
            const top = el.getBoundingClientRect().top + window.scrollY - BAR_OFFSET;
            this.suppressUntil = performance.now() + (reduce ? 150 : 800);
            window.scrollTo({ top: Math.max(0, top), behavior: reduce ? "auto" : "smooth" });
        }
        flash(el, cls) {
            el.classList.remove(cls);
            void el.offsetWidth;
            el.classList.add(cls);
            // 每个元素自己的定时器：连续点两个引用时，不会互相吞掉清理
            window.setTimeout(() => el.classList.remove(cls), 1500);
        }
        showToast(html, ms) {
            this.toastEl.innerHTML = html;
            this.toastEl.hidden = false;
            window.clearTimeout(this.toastTimer);
            this.toastTimer = window.setTimeout(() => this.hideToast(), ms);
        }
        hideToast() {
            this.toastEl.hidden = true;
            this.toastEl.innerHTML = "";
            window.clearTimeout(this.toastTimer);
        }
        // ───────────────────────────── 滚动联动 ─────────────────────────────
        measure() {
            const sy = window.scrollY;
            this.metrics = Array.from(this.articleEl.querySelectorAll(".block[data-rank]"))
                .map((el) => {
                const r = el.getBoundingClientRect();
                return { rank: Number(el.dataset.rank), top: r.top + sy, bottom: r.bottom + sy };
            })
                .filter((m) => Number.isFinite(m.rank))
                .sort((a, b) => a.rank - b.rank);
            const refs = document.getElementById("reader-references");
            this.refsTop = refs ? refs.getBoundingClientRect().top + sy : null;
        }
        /** 阅读行落在哪个正文块上；都落空时取离阅读行最近的可见块，平手中 rank 小者。
         *  双栏里同一行有左右两栏的块，取 rank 小者 = 取左栏，符合阅读顺序。 */
        cursorRank() {
            if (!this.metrics.length)
                return null;
            const line = window.scrollY + READING_LINE;
            let hit = null;
            for (const m of this.metrics) {
                if (m.top <= line && m.bottom >= line && (!hit || m.rank < hit.rank))
                    hit = m;
            }
            if (hit)
                return hit.rank;
            const vTop = window.scrollY + BAR_OFFSET;
            const vBot = window.scrollY + window.innerHeight;
            let best = null;
            let bestD = Infinity;
            for (const m of this.metrics) {
                if (m.bottom <= vTop || m.top >= vBot)
                    continue;
                const d = m.top > line ? m.top - line : line - m.bottom;
                if (d < bestD - 0.5 || (Math.abs(d - bestD) <= 0.5 && (!best || m.rank < best.rank))) {
                    bestD = d;
                    best = m;
                }
            }
            return best ? best.rank : null;
        }
        updateSpy(force) {
            if (!force && performance.now() < this.suppressUntil)
                return;
            const rank = this.cursorRank();
            // 先判文献区：滚进文末后正文块可能全部离开视口（rank === null），
            // 若先 return，References 永远不会高亮。
            const refsNode = this.flatToc.find((n) => n.kind === "references") ?? null;
            if (this.refsTop !== null && refsNode) {
                const line = window.scrollY + READING_LINE;
                // scrollY > 0：整篇不足一屏时，"到底"不代表读到了文末，别把 References 抢先点亮
                const atBottom = window.scrollY > 0 &&
                    window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4;
                if (this.refsTop <= line || atBottom) {
                    this.setActiveToc(refsNode.id, false);
                    return;
                }
            }
            if (rank === null)
                return;
            let active = null;
            for (const n of this.flatToc) {
                if (n.kind === "references")
                    continue;
                const r = this.rankByAnchor.get(n.anchor);
                if (r !== undefined && r <= rank)
                    active = n;
            }
            this.setActiveToc(active ? active.id : null, false);
        }
        setActiveToc(id, userClick) {
            if (id === this.activeTocId && !userClick)
                return;
            this.activeTocId = id;
            let btn = null;
            for (const b of this.tocListEl.querySelectorAll("[data-node]")) {
                const on = b.dataset.node === id;
                b.classList.toggle("is-active", on);
                if (on) {
                    b.setAttribute("aria-current", "location");
                    btn = b;
                }
                else
                    b.removeAttribute("aria-current");
            }
            if (btn) {
                const box = this.tocScrollEl.getBoundingClientRect();
                const r = btn.getBoundingClientRect();
                if (r.top < box.top + 8)
                    this.tocScrollEl.scrollTop -= box.top + 8 - r.top;
                else if (r.bottom > box.bottom - 8)
                    this.tocScrollEl.scrollTop += r.bottom - (box.bottom - 8);
            }
        }
        setMode(mode) {
            if (mode === this.mode)
                return;
            const keep = this.activeTocId;
            this.mode = mode;
            this.root.dataset.mode = mode;
            for (const b of this.root.querySelectorAll(".mode-btn")) {
                b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
            }
            this.renderArticle();
            const node = keep ? this.flatToc.find((n) => n.id === keep) : null;
            requestAnimationFrame(() => {
                this.measure();
                if (node) {
                    const el = document.getElementById(node.anchor);
                    if (el) {
                        const reduce = reduceMotion();
                        this.suppressUntil = performance.now() + 200;
                        window.scrollTo({
                            top: Math.max(0, el.getBoundingClientRect().top + window.scrollY - BAR_OFFSET),
                            behavior: reduce ? "auto" : "smooth",
                        });
                    }
                }
                this.updateSpy(true);
            });
            const anchor = node ? `#${node.anchor}` : "";
            try {
                history.replaceState(null, "", anchor || location.pathname + location.search);
            }
            catch { /* ignore */ }
        }
        // ───────────────────────────── 深链 ─────────────────────────────
        followHash() {
            let raw = "";
            try {
                raw = decodeURIComponent(location.hash.replace(/^#/, ""));
            }
            catch {
                return;
            }
            if (!raw)
                return;
            if (raw.startsWith("occ-")) {
                const occId = raw.slice(4);
                const marker = document.getElementById(raw);
                if (this.occById.has(occId) && marker) {
                    this.measure();
                    this.markOccurrence(occId);
                    this.scrollToEl(marker);
                }
                return;
            }
            const el = document.getElementById(raw);
            if (!el)
                return;
            this.measure();
            const reduce = reduceMotion();
            this.suppressUntil = performance.now() + (reduce ? 150 : 800);
            window.scrollTo({ top: Math.max(0, el.getBoundingClientRect().top + window.scrollY - BAR_OFFSET), behavior: "auto" });
            if (raw.startsWith("ref-")) {
                for (const n of this.root.querySelectorAll(".ref-entry.is-target"))
                    n.classList.remove("is-target");
                el.classList.add("is-target");
            }
            this.updateSpy(true);
        }
    }
    Reader.ReaderView = ReaderView;
})(Reader || (Reader = {}));
