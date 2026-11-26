/* 文献库 / 阅读器 / 引用图谱 / 推荐 / 写作助手
 * 复用 app.js 暴露的 RA（api / el / toast / download / setView）。
 * 所有用户与模型输出一律走 textContent，不拼 innerHTML。
 */
(() => {
  'use strict';

  const RA = window.RA;
  if (!RA) return;
  const { api, el, toast, download, setView } = RA;
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const S = {
    papers: [],
    selected: new Set(),
    pollTimer: 0,
    readerId: '',
    reader: null,
    blocks: [],
    translations: {},
    activePage: 0,
    back: null,
    graph: null,
    lastSelection: '',
    chartFile: null,
  };

  const STATUS_LABEL = {
    queued: '排队中', parsing: '解析中', ready: '已就绪', failed: '失败',
  };
  const VERDICT_LABEL = { duplicate: '同一篇论文的不同版本', related: '内容相关', distinct: '不同论文' };
  const CHANGE_LABEL = { modified: '有修订', added: '新增', removed: '删除' };
  const fmtSize = (n) => {
    if (!n) return '—';
    return n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
  };
  const short = (text, n) => {
    const s = String(text || '').replace(/\s+/g, ' ').trim();
    return s.length > n ? `${s.slice(0, n)}…` : s;
  };

  function emptyNote(title, hint) {
    return el('div', { class: 'empty' }, el('strong', { text: title }), el('span', { text: hint }));
  }

  // 模型答案里的 **加粗** 用 DOM 节点渲染，其余字符原样当文本，杜绝 innerHTML
  function boldText(text) {
    const frag = document.createDocumentFragment();
    const re = /\*\*([^*]+)\*\*/g;
    let last = 0;
    for (let m = re.exec(text); m; m = re.exec(text)) {
      if (m.index > last) frag.append(document.createTextNode(text.slice(last, m.index)));
      frag.append(el('strong', { text: m[1] }));
      last = m.index + m[0].length;
    }
    if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
    return frag;
  }

  function paperTitle(p) {
    return p.title || p.filename || p.id;
  }

  function readyPapers() {
    return S.papers.filter((p) => p.status === 'ready');
  }

  /* ══════════════════════ 文献库 ══════════════════════ */

  async function loadPapers({ quiet = false } = {}) {
    const query = $('#libSearch').value.trim();
    const status = $('#libStatus').value;
    try {
      const data = await api(`/api/papers?query=${encodeURIComponent(query)}&status=${encodeURIComponent(status)}`);
      S.papers = data.items || [];
      const alive = new Set(S.papers.map((p) => p.id));
      for (const id of Array.from(S.selected)) if (!alive.has(id)) S.selected.delete(id);
      const stats = data.stats || {};
      $('#libStats').textContent =
        `共 ${stats.papers || 0} 篇 · ${stats.pages || 0} 页 · ` +
        Object.entries(stats.by_status || {}).map(([k, v]) => `${STATUS_LABEL[k] || k} ${v}`).join(' · ');
      renderPaperList();
      fillPaperSelectors();
      schedulePoll();
    } catch (error) {
      if (!quiet) toast(`文献列表加载失败：${error.message}`, true);
    }
  }

  function schedulePoll() {
    clearTimeout(S.pollTimer);
    const busy = S.papers.some((p) => p.status === 'queued' || p.status === 'parsing');
    if (!busy) return;
    S.pollTimer = setTimeout(() => loadPapers({ quiet: true }), 2500);
  }

  function renderPaperList() {
    const host = $('#libList');
    host.replaceChildren();
    if (!S.papers.length) {
      host.append(emptyNote('文献库为空', '上传 PDF 后自动解析；支持多选与 541 页长文档。'));
      return;
    }
    for (const p of S.papers) host.append(paperCard(p));
  }

  function paperCard(p) {
    const checkbox = el('input', {
      type: 'checkbox',
      'aria-label': `选择 ${short(paperTitle(p), 60)}`,
      onchange: (event) => {
        if (event.target.checked) S.selected.add(p.id); else S.selected.delete(p.id);
        updateSelectionHints();
      },
    });
    checkbox.checked = S.selected.has(p.id);

    const meta = [
      p.year || '年份未知',
      p.page_count ? `${p.page_count} 页` : '',
      fmtSize(p.size_bytes),
      STATUS_LABEL[p.status] || p.status,
      p.arxiv_id || '',
    ].filter(Boolean).join(' · ');

    const actions = el('div', { class: 'row wrap card-actions' });
    if (p.status === 'ready') {
      actions.append(
        el('button', { type: 'button', class: 'ghost small', text: '阅读', onclick: () => openReader(p.id) }),
        el('button', {
          type: 'button', class: 'ghost small', text: '参考文献下载',
          title: '按标题回查 arXiv，只下载标题高度匹配的条目',
          onclick: (event) => downloadReferences(p.id, event.target),
        }),
      );
      const twin = S.papers.find((o) => o.id !== p.id && p.base_id
        && o.base_id === p.base_id && o.status === 'ready');
      if (twin) {
        actions.append(el('button', {
          type: 'button', class: 'ghost small', text: `版本对比 ${twin.version || '另一版'}`,
          title: '逐段比较两个版本的差异',
          onclick: () => compareVersions(p, twin),
        }));
      }
    }
    if (p.status === 'failed') {
      actions.append(el('button', {
        type: 'button', class: 'ghost small', text: '重试',
        onclick: async () => {
          try { await api(`/api/papers/${p.id}/retry`, { method: 'POST' }); loadPapers(); }
          catch (error) { toast(error.message, true); }
        },
      }));
    }
    actions.append(
      el('button', {
        type: 'button', class: 'ghost small', text: '重命名',
        onclick: async () => {
          const title = window.prompt('新的标题', paperTitle(p));
          if (!title || title === paperTitle(p)) return;
          try {
            await api(`/api/papers/${p.id}`, { method: 'PATCH', body: JSON.stringify({ title }) });
            loadPapers();
          } catch (error) { toast(error.message, true); }
        },
      }),
      el('button', {
        type: 'button', class: 'ghost small danger', text: '删除',
        onclick: async () => {
          if (!window.confirm(`删除《${short(paperTitle(p), 50)}》？原文件与解析结果都会删除。`)) return;
          try {
            await api(`/api/papers/${p.id}`, { method: 'DELETE' });
            S.selected.delete(p.id);
            loadPapers();
          } catch (error) { toast(error.message, true); }
        },
      }),
    );

    const body = el('div', { class: 'paper-card-body' },
      el('div', { class: 'paper-title-row' },
        el('label', { class: 'paper-check' }, checkbox),
        el('button', {
          type: 'button', class: 'paper-title', text: paperTitle(p),
          onclick: () => { if (p.status === 'ready') openReader(p.id); },
        }),
      ),
      el('div', { class: 'paper-meta', text: meta }),
      p.abstract ? el('p', { class: 'paper-abstract', text: short(p.abstract, 220) }) : null,
      p.error ? el('p', { class: 'warning', text: p.error }) : null,
      p.status === 'parsing' || p.status === 'queued'
        ? el('div', { class: 'progress' }, el('span', { style: `width:${Math.round((p.progress || 0) * 100)}%` }))
        : null,
      actions,
    );
    return el('article', { class: 'panel paper-card' }, body);
  }

  function updateSelectionHints() {
    const n = S.selected.size;
    $('#qaSelected').textContent = n ? `已选择 ${n} 篇文献` : '未选择文献时使用全部已就绪文献';
    $$('.graph-paper-check').forEach((box) => {
      if (box.dataset.paperId) box.checked = S.selected.has(box.dataset.paperId);
    });
  }

  async function uploadPapers(files) {
    if (!files || !files.length) return;
    const form = new FormData();
    for (const f of files) form.append('files', f);
    const notes = $('#libUploadNotes');
    notes.replaceChildren(emptyNote('上传中…', `${files.length} 个文件正在写入并排队解析。`));
    try {
      const data = await api('/api/papers/upload', { method: 'POST', body: form });
      notes.replaceChildren();
      for (const item of data.items || []) notes.append(uploadNote(item));
      await loadPapers();
    } catch (error) {
      notes.replaceChildren(el('p', { class: 'warning', text: `上传失败：${error.message}` }));
    }
  }

  function uploadNote(item) {
    if (item.error) {
      return el('div', { class: 'upload-note error' },
        el('strong', { text: item.filename }), el('span', { text: item.error }));
    }
    const paper = item.paper || {};
    if (item.created) {
      return el('div', { class: 'upload-note' },
        el('strong', { text: item.filename }),
        el('span', { text: `已入库，解析任务 #${paper.id}` }));
    }
    const box = el('div', { class: 'upload-note warn' },
      el('strong', { text: item.filename }),
      el('span', { text: '库中已有同一文件；未重复解析。' }));
    for (const dup of item.duplicates || []) {
      const old = dup.paper || {};
      const row = el('div', { class: 'dup-row' },
        el('span', {
          text: `${dup.relation === 'version' ? '同一 arXiv 论文的其他版本' : '标题高度相似'}：` +
            `《${short(old.title || old.filename || old.id, 60)}》` +
            (old.version ? `（${old.version}）` : ''),
        }),
        el('button', {
          type: 'button', class: 'ghost small', text: '用新文件替换旧版本',
          onclick: async (event) => {
            if (!window.confirm('替换会删除旧版本的解析结果，确定继续？')) return;
            event.target.disabled = true;
            try {
              await api(`/api/papers/${paper.id}/replace`, {
                method: 'POST', body: JSON.stringify({ old_id: old.id }),
              });
              box.replaceChildren(el('span', { text: `已用新文件替换《${short(old.title || old.id, 60)}》。` }));
              loadPapers();
            } catch (error) {
              event.target.disabled = false;
              toast(error.message, true);
            }
          },
        }),
        el('button', {
          type: 'button', class: 'ghost small', text: '保留两者',
          onclick: () => { box.classList.remove('warn'); row.remove(); },
        }),
      );
      box.append(row);
    }
    return box;
  }

  async function downloadArxiv() {
    const raw = $('#libArxiv').value.trim();
    if (!raw) { toast('请输入 arXiv ID 或标题', true); return; }
    const btn = $('#libDownload');
    btn.disabled = true;
    btn.textContent = '处理中…';
    try {
      const isId = /^\d{4}\.\d{4,5}(v\d+)?$/.test(raw);
      const onlySearch = $('#libSearchOnly').checked;
      const payload = isId
        ? { arxiv_id: raw }
        : { query: raw, max_results: 5, download: !onlySearch };
      const data = await api('/api/papers/download', { method: 'POST', body: JSON.stringify(payload) });
      const notes = $('#libUploadNotes');
      notes.replaceChildren();
      if (isId) {
        notes.append(el('div', { class: 'upload-note' },
          el('strong', { text: data.arxiv || raw }),
          el('span', { text: data.created ? `已下载并入库 #${data.paper.id}` : (data.note || '库中已有该论文') })));
      } else {
        const list = el('div', { class: 'upload-note' });
        list.append(el('strong', { text: `检索：${data.query || raw}` }));
        for (const c of data.candidates || []) {
          list.append(el('div', { class: 'dup-row' },
            el('span', { text: `${c.arxiv_id} · ${short(c.title, 80)}` }),
            el('button', {
              type: 'button', class: 'ghost small', text: '下载',
              onclick: async (event) => {
                event.target.disabled = true;
                try {
                  await api('/api/papers/download', {
                    method: 'POST', body: JSON.stringify({ arxiv_id: c.arxiv_id }),
                  });
                  loadPapers();
                } catch (error) { event.target.disabled = false; toast(error.message, true); }
              },
            })));
        }
        if (!(data.candidates || []).length) list.append(el('span', { text: '没有找到匹配结果。' }));
        notes.append(list);
      }
      await loadPapers();
    } catch (error) {
      toast(error.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = '下载 / 检索';
    }
  }

  async function compareVersions(a, b) {
    const host = $('#libCompare');
    const vnum = (p) => { const m = /(\d+)$/.exec(p.version || ''); return m ? Number(m[1]) : -1; };
    if (vnum(a) < 0 || (vnum(b) >= 0 && vnum(a) > vnum(b))) [a, b] = [b, a];
    host.hidden = false;
    host.replaceChildren(emptyNote('对比中…', '基于两篇文献的解析结果逐段比较。'));
    try {
      const d = await api(`/api/papers/${a.id}/compare/${b.id}`);
      const detail = (d.diff_detail || []).filter((x) => x.change && x.change !== 'unchanged');
      const nodes = [
        el('h3', { text: '版本对比' }),
        el('p', { class: 'hint', text: `${short(paperTitle(b), 60)} · ${a.version || '旧版'} → ${b.version || '未标版本'}` }),
        el('p', { class: 'qa-text' },
          el('strong', { text: `${VERDICT_LABEL[d.verdict] || d.verdict} · 相似度 ${Math.round((d.similarity || 0) * 100)}%` })),
        d.version_advice ? el('p', { class: 'qa-text', text: d.version_advice }) : null,
        d.diff_summary ? el('p', { class: 'hint', text: d.diff_summary.split('\n')[0] }) : null,
        detail.length ? el('h4', { text: `差异段落（${detail.length}）` }) : null,
        ...detail.slice(0, 20).map((x) => el('article', { class: 'evidence' },
          el('strong', { text: `${x.heading || '正文'} · ${CHANGE_LABEL[x.change] || x.change}` }),
          el('p', { class: 'hint', text: short(x.summary || '', 300) }))),
      ];
      host.replaceChildren(...nodes.filter(Boolean));
      host.scrollIntoView({ block: 'start', behavior: 'smooth' });
    } catch (error) {
      host.replaceChildren(el('p', { class: 'warning', text: error.message }));
    }
  }

  async function downloadReferences(paperId, btn) {
    btn.disabled = true;
    btn.textContent = '回查中…';
    try {
      const data = await api(`/api/papers/${paperId}/references/download?limit=5`, { method: 'POST' });
      const rows = (data.results || []).map((r) =>
        `${r.status === 'downloaded' ? '✓' : r.status === 'existing' ? '已有' : r.status === 'error' ? '✗' : '未找到'} ` +
        `${r.arxiv_id || ''} ${short(r.title || '', 70)}`).join('\n');
      toast(`参考文献回查：尝试 ${data.attempted} 条，跳过低价值 ${data.skipped_low_value} 条，下载 ${(data.results || []).filter((r) => r.created).length} 篇。`);
      if (rows) $('#libUploadNotes').replaceChildren(el('pre', { class: 'code-out', text: rows }));
      loadPapers();
    } catch (error) {
      toast(error.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = '参考文献下载';
    }
  }

  /* ── 文献问答 ── */
  async function askPapers(event) {
    event.preventDefault();
    const query = $('#qaQuery').value.trim();
    if (!query) { toast('请输入问题', true); return; }
    const btn = $('#qaAsk');
    btn.disabled = true;
    btn.textContent = '检索中…';
    const out = $('#qaAnswer');
    out.replaceChildren(emptyNote('检索原文证据…', '只使用库内已解析文献。'));
    try {
      const data = await api('/api/qa', {
        method: 'POST',
        body: JSON.stringify({ query, paper_ids: Array.from(S.selected), top_k: 6 }),
      });
      out.replaceChildren();
      const answer = data.answer || '（没有生成摘要，见下方证据原文）';
      const body = el('div', { class: 'qa-text' });
      body.append(boldText(answer));   // 只处理 **加粗**，其余按纯文本渲染
      out.append(body);
      for (const w of data.warnings || []) out.append(el('p', { class: 'warning', text: w }));
      const evidence = data.evidence || [];
      if (evidence.length) {
        out.append(el('h4', { text: '原文证据' }));
        evidence.forEach((e, i) => {
          const row = el('article', { class: 'evidence' },
            el('div', { class: 'evidence-head' },
              el('strong', { text: `[E${i + 1}] ${short(e.title || e.paper_title || e.paper_id, 60)}` }),
              el('span', { class: 'hint', text: `${e.section || '正文'} · 第 ${e.page || '?'} 页` })),
            el('blockquote', { text: e.quote || '' }),
            el('button', {
              type: 'button', class: 'ghost small', text: '定位原文',
              onclick: () => openReader(e.paper_id, e.page, e.block_id),
            }));
          out.append(row);
        });
      }
    } catch (error) {
      out.replaceChildren(el('p', { class: 'warning', text: `问答失败：${error.message}` }));
    } finally {
      btn.disabled = false;
      btn.textContent = '提问';
    }
  }

  /* ══════════════════════ 阅读器 ══════════════════════ */

  function fillPaperSelectors() {
    const select = $('#readerPaper');
    const current = S.readerId || select.value;
    select.replaceChildren();
    for (const p of readyPapers()) {
      select.append(el('option', { value: p.id, text: short(paperTitle(p), 80) }));
    }
    if (current && readyPapers().some((p) => p.id === current)) select.value = current;
    renderGraphPaperPicker();
  }

  async function openReader(paperId, page, blockId) {
    if (!S.papers.length) loadPapers({ quiet: true });
    setView('reader');
    $('#readerPaper').value = paperId;
    if (paperId === S.readerId && S.reader) {
      jumpTo(page || 1, blockId);
      return;
    }
    S.readerId = paperId;
    S.reader = null;
    S.blocks = [];
    S.translations = {};
    S.activePage = 0;
    await loadReader(page, blockId);
  }

  async function loadReader(page, blockId) {
    const pid = S.readerId;
    if (!pid) {
      $('#readerTitle').textContent = '未选择文献';
      $('#readerMeta').textContent = '在文献库点击“阅读”打开';
      return;
    }
    $('#readerBlocks').replaceChildren(emptyNote('加载解析结果…', '首次解析完成后才能阅读。'));
    try {
      const data = await api(`/api/papers/${pid}/reader`);
      S.reader = data;
      S.blocks = data.blocks || [];
      S.translations = data.translations || {};
      const meta = data.paper || {};
      $('#readerTitle').textContent = data.meta?.title || meta.title || pid;
      updateReaderMeta();
      $('#readerWarn').hidden = !(data.warnings || []).length;
      $('#readerWarn').textContent = (data.warnings || []).join('；');
      for (const [label, render] of [
        ['目录', () => renderToc(data.toc || [])],
        ['正文', () => renderBlocks()],
        ['参考文献', () => renderReferences(data.references || [])],
        ['定位', () => jumpTo(page || 1, blockId)],
      ]) {
        try { render(); } catch (error) { throw new Error(`${label}渲染失败：${error.message}`); }
      }
    } catch (error) {
      $('#readerBlocks').replaceChildren(el('p', { class: 'warning', text: error.message }));
      if (/尚未解析/.test(error.message)) {
        setTimeout(() => { if (S.readerId === pid) loadReader(page, blockId); }, 2500);
      }
    }
  }

  function updateReaderMeta() {
    const r = S.reader;
    if (!r) return;
    const stats = r.stats || {};
    $('#readerMeta').textContent =
      `${stats.pages || 0} 页 · ${stats.blocks || 0} 块 · ${stats.references || 0} 条参考文献 · ` +
      `${stats.citations || 0} 处引用 · 已译 ${Object.keys(S.translations).length} 块`;
  }

  function renderToc(nodes, host, depth = 0) {
    host = host || $('#readerToc');
    if (depth === 0) host.replaceChildren();
    if (!nodes.length && depth === 0) {
      host.append(el('p', { class: 'hint', text: '没有识别到目录。' }));
      return;
    }
    const list = el('ul', { class: depth ? 'toc-children' : 'toc-root' });
    for (const node of nodes) {
      const item = el('li', {},
        el('button', {
          type: 'button', class: 'toc-link',
          style: `padding-left:${6 + depth * 12}px`,
          text: `${node.number ? node.number + ' ' : ''}${short(node.text, 70)}`,
          title: node.text,
          onclick: () => {
            if (node.kind === 'references') {
              const refs = document.querySelector('.reader-refs');
              if (refs) scrollBlock(refs, false);
              return;
            }
            jumpTo(node.page, node.anchor?.replace('blk-', ''));
          },
        }));
      if ((node.children || []).length) item.append(...renderTocChildren(node.children, depth + 1));
      list.append(item);
    }
    host.append(list);
  }

  function renderTocChildren(nodes, depth) {
    const holder = el('div');
    renderToc(nodes, holder, depth);
    return Array.from(holder.children);
  }

  const FIGURE_TYPES = new Set(['figure', 'table', 'formula', 'title', 'heading', 'reference']);

  function renderBlocks() {
    const host = $('#readerBlocks');
    const scroll = host.parentElement?.scrollTop || 0;
    host.replaceChildren();
    if (!S.blocks.length) {
      host.append(emptyNote('没有可显示的正文', '解析结果为空或仍在处理。'));
      return;
    }
    for (const block of S.blocks) host.append(blockNode(block));
    host.parentElement.scrollTop = scroll;
  }

  function blockNode(block) {
    const type = block.type || 'paragraph';
    const id = block.id || '';
    let node;
    if (type === 'heading' || type === 'title') {
      const level = Math.min(5, Math.max(2, (block.level || 1) + 1));
      node = el(`h${level}`, { class: 'reader-heading' });
      node.append(...inlineSegments(block));
    } else if (type === 'formula') {
      node = block.latex || block.text
        ? el('pre', { class: 'reader-formula', text: block.latex || block.text })
        : imageBlock(block, '公式');
    } else if (type === 'table') {
      node = el('div', { class: 'reader-table' });
      if (block.markdown) node.append(el('pre', { text: block.markdown }));
      else if (block.text) node.append(el('pre', { text: block.text }));
      if (block.caption) node.append(el('p', { class: 'figure-caption', text: block.caption }));
    } else if (type === 'figure') {
      node = imageBlock(block, '插图');
    } else if (type === 'list_item') {
      node = el('p', { class: 'reader-list-item' });
      node.append(el('span', { class: 'list-bullet', text: '• ' }), ...inlineSegments(block));
    } else if (!(block.text || '').trim() && block.image_path) {
      node = imageBlock(block, '图片');
    } else {
      node = el('p', { class: 'reader-paragraph' });
      node.append(...inlineSegments(block));
    }
    node.classList.add(`block-${type}`);
    if (id) node.id = `blk-${id}`;
    if (block.page) node.dataset.page = String(block.page);
    if (id) node.dataset.blockId = id;

    const zh = (S.translations[id] || {}).zh;
    if ($('#readerBilingual').checked && zh) {
      node.append(el('div', { class: 'block-translation', text: zh }));
    }
    return node;
  }

  function imageBlock(block, label) {
    const figure = el('figure', { class: 'reader-figure' });
    if (block.image_path) {
      figure.append(el('img', {
        src: `/api/papers/${S.readerId}/asset/${encodeURIComponent(block.image_path)}`,
        alt: block.caption || label, loading: 'lazy',
        onerror: (event) => { event.target.replaceWith(el('p', { class: 'hint', text: '图片无法加载。' })); },
      }));
    }
    const cap = block.caption || block.text;
    if (cap) figure.append(el('figcaption', { text: cap }));
    return figure;
  }

  function inlineSegments(block) {
    const segments = block.segments;
    if (!segments || !segments.length) return [document.createTextNode(block.text || '')];
    return segments.map((seg) => {
      if (seg.kind !== 'cite') return document.createTextNode(seg.text || '');
      return el('button', {
        type: 'button', class: `cite ${seg.status || 'missing'}`,
        text: seg.text || '',
        title: seg.targets?.length ? '跳转到参考文献' : '参考文献表中没有该编号',
        onclick: () => citeJump(block, seg),
      });
    });
  }

  function citeJump(block, seg) {
    const target = (seg.targets || [])[0];
    if (!target) { toast('参考文献表中没有该编号', true); return; }
    S.back = { page: block.page, blockId: block.id };
    const ref = document.getElementById(`ref-${target}`) || document.getElementById(target);
    if (ref) {
      const details = ref.closest('details');
      if (details) details.open = true;   // 折叠状态下没法滚动到条目
      requestAnimationFrame(() => {
        scrollBlock(ref, true);
        ref.classList.add('flash');
        setTimeout(() => ref.classList.remove('flash'), 1600);
      });
      if (block.page) syncPdfPage(Number(block.page));
      return;
    }
    const item = (S.reader?.references || []).find((r) => r.id === target);
    if (item && item.page) jumpTo(item.page);
  }

  function renderReferences(refs) {
    const host = $('#readerRefs');
    host.replaceChildren();
    if (!refs.length) {
      host.append(el('li', { class: 'hint', text: '没有识别到参考文献。' }));
      return;
    }
    for (const ref of refs) {
      const item = el('li', { id: `ref-${ref.id}` },
        el('span', { class: 'ref-num', text: ref.number ? `[${ref.number}]` : '' }),
        el('span', { text: ref.text || '' }),
      );
      const back = ref.cite_ids && ref.cite_ids.length
        ? el('button', {
          type: 'button', class: 'ghost small', text: `返回引用处 (${ref.cite_ids.length})`,
          onclick: () => backToCite(ref),
        })
        : null;
      if (back) item.append(back);
      if (ref.url) item.append(el('a', { href: ref.url, target: '_blank', rel: 'noopener', text: '链接' }));
      if (ref.arxiv) item.append(el('span', { class: 'hint', text: ` arXiv:${ref.arxiv}` }));
      host.append(item);
    }
  }

  function backToCite(ref) {
    const occId = (ref.cite_ids || [])[0];
    const occ = (S.reader?.citations?.occurrences || []).find((o) => o.id === occId);
    if (!occ) return;
    jumpTo(undefined, occ.block_id);
  }

  function scrollBlock(node, center) {
    // 只滚动正文面板；scrollIntoView 会连带整页滚动，把页头挤出去
    const scroller = node.closest('.reader-text');
    if (!scroller) return;
    const top = node.getBoundingClientRect().top - scroller.getBoundingClientRect().top
      + scroller.scrollTop - (center ? scroller.clientHeight / 3 : 12);
    scroller.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
  }

  function jumpTo(page, blockId) {
    if (blockId) {
      const node = document.getElementById(`blk-${blockId}`);
      if (node) {
        scrollBlock(node, true);
        node.classList.add('flash');
        setTimeout(() => node.classList.remove('flash'), 1600);
        const p = Number(node.dataset.page || page || 0);
        if (p) { S.activePage = p; syncPdfPage(p); }
        return;
      }
    }
    if (page) {
      S.activePage = Number(page);
      syncPdfPage(Number(page));
      const first = S.blocks.find((b) => Number(b.page) === Number(page));
      if (first) {
        const node = document.getElementById(`blk-${first.id}`);
        if (node) scrollBlock(node, false);
      }
    }
  }

  function syncPdfPage(page) {
    if (!page || !S.readerId) return;
    const src = `/api/papers/${S.readerId}/file#page=${page}&zoom=page-width`;
    const frame = $('#readerPdf');
    if (frame.dataset.page === String(page)) return;
    frame.dataset.page = String(page);
    frame.src = src;
  }

  let followTimer = 0;
  function onReaderScroll() {
    if (!$('#readerFollow').checked) return;
    clearTimeout(followTimer);
    followTimer = setTimeout(() => {
      const pane = $('#readerText');
      const top = pane.getBoundingClientRect().top;
      const nodes = Array.from($('#readerBlocks').children);
      const visible = nodes.find((node) => node.getBoundingClientRect().bottom > top + 24);
      const page = visible && Number(visible.dataset.page);
      if (page) { S.activePage = page; syncPdfPage(page); }
    }, 400);
  }

  async function translateCurrentPage() {
    if (!S.readerId || !S.blocks.length) { toast('先打开一篇文献', true); return; }
    const page = S.activePage || Number(S.blocks[0].page) || 1;
    await translatePages([page], `第 ${page} 页`, $('#readerTranslate'));
  }

  async function translateAll() {
    if (!S.readerId || !S.blocks.length) { toast('先打开一篇文献', true); return; }
    const pages = Array.from(new Set(S.blocks.map((b) => Number(b.page || 0)).filter(Boolean))).sort((a, b) => a - b);
    const todo = pages.filter((page) => S.blocks.some((b) =>
      Number(b.page) === page && (b.type === 'paragraph' || b.type === 'heading' || b.type === 'list_item' || b.type === 'title')));
    if (!todo.length) return;
    if (!window.confirm(`将逐页调用翻译模型处理 ${todo.length} 页，耗时与费用取决于篇幅。继续？`)) return;
    await translatePages(todo, '全文', $('#readerTranslateAll'));
  }

  async function translatePages(pages, label, btn) {
    const other = btn.id === 'readerTranslate' ? $('#readerTranslateAll') : $('#readerTranslate');
    btn.disabled = other.disabled = true;
    const original = btn.textContent;
    let failed = 0;
    try {
      for (let i = 0; i < pages.length; i += 1) {
        btn.textContent = `${label} ${i + 1}/${pages.length}…`;
        try {
          const data = await api(`/api/papers/${S.readerId}/translate`, {
            method: 'POST',
            body: JSON.stringify({ page: pages[i], limit: 40 }),
          });
          Object.assign(S.translations, data.translations || {});
          for (const w of data.warnings || []) toast(w, true);
        } catch (error) {
          failed += 1;
          toast(error.message, true);
          break;
        }
        updateReaderMeta();
      }
      renderBlocks();
    } finally {
      btn.disabled = other.disabled = false;
      btn.textContent = original;
      if (label !== '全文') toast(failed ? '翻译失败' : '本页翻译完成');
      else toast(failed ? '全文翻译中断' : `已翻译 ${pages.length} 页`);
    }
  }

  function showSelectionButton() {
    const pop = $('#readerSelection');
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : '';
    if (!text || text.length > 6000) { pop.hidden = true; return; }
    const anchor = sel.anchorNode;
    if (!anchor || !$('#readerText').contains(anchor)) { pop.hidden = true; return; }
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    S.lastSelection = text;
    const result = pop.querySelector('.selection-result');
    if (result) result.remove();
    pop.style.left = `${Math.min(window.innerWidth - 240, Math.max(12, rect.left))}px`;
    pop.style.top = `${Math.max(12, rect.top - 46)}px`;
    pop.hidden = false;
  }

  async function translateSelection() {
    if (!S.readerId || !S.lastSelection) return;
    const pop = $('#readerSelection');
    const btn = $('#readerSelectionBtn');
    btn.disabled = true;
    try {
      const data = await api(`/api/papers/${S.readerId}/translate-selection`, {
        method: 'POST', body: JSON.stringify({ text: S.lastSelection }),
      });
      const box = el('div', { class: 'selection-result', text: data.zh || '（空译文）' });
      pop.append(box);
    } catch (error) {
      pop.append(el('div', { class: 'selection-result warning', text: error.message }));
    } finally {
      btn.disabled = false;
    }
  }

  async function exportMarkdown() {
    if (!S.readerId) return;
    const lang = $('#readerLang').value;
    try {
      const data = await api(`/api/papers/${S.readerId}/markdown?lang=${encodeURIComponent(lang)}`);
      const title = (S.reader?.meta?.title || S.readerId).replace(/[\\/:*?"<>|]/g, '_');
      download(`${title}.md`, data.markdown || '', 'text/markdown;charset=utf-8');
    } catch (error) {
      toast(`导出失败：${error.message}`, true);
    }
  }

  /* ══════════════════════ 引用图谱 ══════════════════════ */

  function renderGraphPaperPicker() {
    const host = $('#graphPapers');
    host.replaceChildren();
    const ready = readyPapers();
    if (!ready.length) {
      host.append(el('span', { class: 'hint', text: '没有已解析文献。' }));
      return;
    }
    for (const p of ready) {
      const box = el('input', {
        type: 'checkbox', class: 'graph-paper-check', dataset: { paperId: p.id },
        onchange: (event) => {
          if (event.target.checked) S.selected.add(p.id); else S.selected.delete(p.id);
          updateSelectionHints();
        },
      });
      box.checked = S.selected.has(p.id);
      host.append(el('label', { class: 'check-item', title: paperTitle(p) },
        box, el('span', { text: short(paperTitle(p), 60) })));
    }
  }

  function selectedPaperIds() {
    return Array.from(S.selected);
  }

  async function buildGraph() {
    let ids = selectedPaperIds();
    if (!ids.length) ids = readyPapers().map((p) => p.id);   // 未勾选时用全部已解析文献
    if (ids.length < 2) { toast('至少勾选 2 篇已解析文献', true); return; }
    const btn = $('#graphBuild');
    btn.disabled = true;
    try {
      const data = await api('/api/graph', { method: 'POST', body: JSON.stringify({ paper_ids: ids }) });
      S.graph = data;
      renderGraph(data);
    } catch (error) {
      toast(error.message, true);
    } finally {
      btn.disabled = false;
    }
  }

  function graphLayout(nodes) {
    const byComm = new Map();
    for (const n of nodes) {
      const key = n.community ?? 0;
      if (!byComm.has(key)) byComm.set(key, []);
      byComm.get(key).push(n);
    }
    const comms = Array.from(byComm.keys());
    const pos = new Map();
    const width = 900, height = 600;
    const outerR = Math.min(width, height) / 2 - 90;
    comms.forEach((comm, i) => {
      const members = byComm.get(comm);
      const angle = (2 * Math.PI * i) / Math.max(1, comms.length) - Math.PI / 2;
      const cx = width / 2 + (comms.length > 1 ? Math.cos(angle) * outerR * 0.55 : 0);
      const cy = height / 2 + (comms.length > 1 ? Math.sin(angle) * outerR * 0.55 : 0);
      const radius = comms.length === 1 ? Math.min(220, 120 + members.length * 8) : Math.min(150, 54 + members.length * 12);
      members.forEach((n, j) => {
        const a = (2 * Math.PI * j) / Math.max(1, members.length) - Math.PI / 2;
        pos.set(n.id, {
          x: cx + Math.cos(a) * radius,
          y: cy + Math.sin(a) * radius,
          r: Math.max(10, Math.min(30, 8 + (n.size || 14) * 0.45)),
          node: n,
        });
      });
    });
    return { pos, width, height };
  }

  function renderGraph(data) {
    const svg = $('#graphSvg');
    svg.replaceChildren();
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    const { pos, width, height } = graphLayout(nodes);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');

    const defs = svgEl('defs');
    const marker = svgEl('marker', {
      id: 'arrow', viewBox: '0 0 10 10', refX: '10', refY: '5',
      markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse',
    });
    marker.append(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#8aa39d' }));
    defs.append(marker);
    svg.append(defs);

    for (const edge of edges) {
      const a = pos.get(edge.source), b = pos.get(edge.target);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const len = Math.hypot(dx, dy) || 1;
      const sx = a.x + (dx / len) * a.r, sy = a.y + (dy / len) * a.r;
      const tx = b.x - (dx / len) * b.r, ty = b.y - (dy / len) * b.r;
      svg.append(svgEl('line', {
        x1: sx, y1: sy, x2: tx, y2: ty,
        stroke: edge.cross_community ? '#c9a227' : '#8aa39d',
        'stroke-width': Math.max(1, Math.min(4, edge.weight / 2)),
        'stroke-dasharray': edge.cross_community ? '5 4' : '',
        'marker-end': 'url(#arrow)', opacity: '0.7',
      }));
    }

    for (const node of nodes) {
      const p = pos.get(node.id);
      if (!p) continue;
      const group = svgEl('g', { class: `graph-node role-${node.role || 'normal'}`, tabindex: '0', role: 'button' });
      group.append(svgEl('title', { text: `${node.title}（${node.role === 'keystone' ? '基石' : node.role === 'peripheral' ? '边缘衍生' : '普通'}）` }));
      group.append(svgEl('circle', {
        cx: p.x, cy: p.y, r: p.r, fill: node.color || '#5b8c7f',
        stroke: node.role === 'keystone' ? '#0f6b5c' : node.role === 'peripheral' ? '#c9a227' : '#ffffff',
        'stroke-width': node.role === 'normal' ? 1.5 : 3,
      }));
      const label = svgEl('text', {
        x: p.x, y: p.y + p.r + 13, 'text-anchor': 'middle', 'font-size': '10', fill: '#10221f',
        'pointer-events': 'none',
      });
      label.textContent = short(node.title || node.label || node.id, 20);
      group.append(label);
      group.addEventListener('click', () => showGraphNode(node, data));
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); showGraphNode(node, data); }
      });
      svg.append(group);
    }

    const meta = data.meta || {};
    $('#graphStats').textContent =
      `${meta.n_nodes || 0} 个节点 · ${meta.n_edges || 0} 条引用边 · ${meta.n_communities || 0} 个社区 · ` +
      `模块度 ${meta.modularity ?? '—'} · 库内匹配 ${data.library?.matched_refs || 0} 条 / 未匹配 ${data.library?.unmatched_refs || 0} 条`;
    renderKeystone(data.keystone || []);
    renderPeripheral(data.peripheral || []);
    renderCommunities(meta.communities || []);
    if ((meta.warnings || []).length) {
      const box = $('#graphOutput');
      box.hidden = false;
      box.replaceChildren(el('h3', { text: '图谱提示' }),
        ...meta.warnings.map((w) => el('p', { class: 'warning', text: w })));
    }
  }

  function svgEl(tag, attrs = {}) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === 'class') node.setAttribute('class', v);
      else if (k === 'text') node.textContent = v;
      else node.setAttribute(k, String(v));
    }
    return node;
  }

  function showGraphNode(node, data) {
    const info = $('#graphNodeInfo');
    info.replaceChildren(
      el('strong', { text: node.title }),
      el('div', { class: 'hint', text: `${node.year || '年份未知'} · ${node.group || ''} · ${node.role}` }),
      el('div', { class: 'hint', text: `PageRank ${node.pagerank} · 介数 ${node.betweenness} · 入度 ${node.in_degree} / 出度 ${node.out_degree}` }),
      el('button', {
        type: 'button', class: 'ghost small', text: '打开阅读器',
        onclick: () => openReader(node.id),
      }),
    );
    const ks = (data.keystone || []).find((k) => k.id === node.id);
    if (ks) info.append(el('p', { class: 'hint', text: ks.why }));
    const per = (data.peripheral || []).find((k) => k.id === node.id);
    if (per) info.append(el('p', { class: 'hint', text: per.why }));
  }

  function renderKeystone(items) {
    const host = $('#graphKeystone');
    host.replaceChildren();
    if (!items.length) host.append(el('p', { class: 'hint', text: '没有基石论文（可能没有边）。' }));
    for (const item of items) {
      host.append(el('div', { class: 'rank-row' },
        el('strong', { text: `#${item.rank} ${short(item.title, 50)}` }),
        el('span', { class: 'hint', text: `综合分 ${item.score} · ${item.group}` }),
        el('p', { class: 'hint', text: item.why })));
    }
  }

  function renderPeripheral(items) {
    const host = $('#graphPeripheral');
    host.replaceChildren();
    if (!items.length) host.append(el('p', { class: 'hint', text: '没有满足边缘衍生规则的论文。' }));
    for (const item of items) {
      host.append(el('div', { class: 'rank-row' },
        el('strong', { text: `#${item.rank} ${short(item.title, 50)}` }),
        el('span', { class: 'hint', text: `创新关联分 ${item.score} · ${item.group}` }),
        el('p', { class: 'hint', text: item.why })));
    }
  }

  function renderCommunities(items) {
    const host = $('#graphCommunities');
    host.replaceChildren();
    for (const c of items) {
      host.append(el('div', { class: 'rank-row' },
        el('span', { class: 'comm-dot', style: `background:${c.color}` }),
        el('strong', { text: `${c.label}（${c.size} 篇）` })));
    }
  }

  async function graphAction(kind) {
    const ids = selectedPaperIds();
    if (!ids.length) { toast('先勾选文献', true); return; }
    const out = $('#graphOutput');
    out.hidden = false;
    out.replaceChildren(emptyNote('生成中…', '基于库内摘要与原文，不引入材料外内容。'));
    try {
      const data = await api(`/api/graph/${kind}`, { method: 'POST', body: JSON.stringify({ paper_ids: ids }) });
      out.replaceChildren();
      if (kind === 'survey') {
        out.append(el('h2', { text: '综述' }), el('pre', { class: 'markdown-text', text: data.markdown || '' }));
        if ((data.issues || []).length) {
          out.append(el('h3', { text: '校验问题' }),
            ...data.issues.map((i) => el('p', { class: 'warning', text: typeof i === 'string' ? i : JSON.stringify(i) })));
        }
      } else {
        const open = data.open_directions || [];
        const resolved = data.resolved || [];
        out.append(el('h2', { text: `Future Work（${open.length} 个开放方向）` }));
        for (const item of open) {
          out.append(el('article', { class: 'rank-row' },
            el('strong', { text: item.idea }),
            el('p', { class: 'hint', text: `来源：${item.source || '—'}` }),
            el('p', { class: 'hint', text: `为何仍未解决：${item.why_open || '—'}` }),
            el('p', { class: 'hint', text: `可行切入点：${item.angle || '—'}` })));
        }
        if (resolved.length) {
          out.append(el('h3', { text: '已被后续工作解决' }));
          for (const item of resolved) {
            out.append(el('div', { class: 'rank-row' },
              el('strong', { text: item.idea }), el('p', { class: 'hint', text: item.solved_by })));
          }
        }
      }
    } catch (error) {
      out.replaceChildren(el('p', { class: 'warning', text: error.message }));
    }
  }

  /* ══════════════════════ 推荐 ══════════════════════ */

  async function loadRecommend({ refresh = false } = {}) {
    if (refresh) {
      const btn = $('#recRefresh');
      btn.disabled = true;
      btn.textContent = '拉取中…';
      try {
        const data = await api('/api/recommend/refresh', {
          method: 'POST',
          body: JSON.stringify({
            days: Number($('#recDays').value) || 14,
            top: Number($('#recTop').value) || 5,
            include_seen: $('#recSeen').checked,
            use_github: $('#recGithub').checked,
          }),
        });
        renderRecommend(data);
      } catch (error) {
        toast(`推荐刷新失败：${error.message}`, true);
      } finally {
        btn.disabled = false;
        btn.textContent = '刷新推荐';
      }
      return;
    }
    try {
      const data = await api('/api/recommend');
      if (data.latest) renderRecommend(data.latest);
      else $('#recList').replaceChildren(emptyNote('还没有推荐结果', '点击“刷新推荐”从 arXiv 拉取候选。'));
    } catch (error) {
      toast(error.message, true);
    }
  }

  function renderRecommend(data) {
    const host = $('#recList');
    host.replaceChildren();
    const meta = data.meta || {};
    $('#recMeta').textContent =
      `生成于 ${meta.generated_at || '—'} · 候选 ${meta.candidates ?? 0} 篇 · 返回 ${meta.returned ?? 0} 篇` +
      (meta.library ? ` · 文献库 ${meta.library.docs} 篇` : '');
    for (const w of meta.warnings || []) host.append(el('p', { class: 'warning', text: w }));
    const items = data.recommendations || [];
    if (!items.length) {
      host.append(emptyNote('没有推荐', '可能是近期候选都用完了；可勾选“包含已推荐”。'));
      return;
    }
    for (const item of items) {
      const repro = item.reproducibility || {};
      const card = el('article', { class: 'panel paper-card' },
        el('div', { class: 'paper-title-row' },
          el('strong', { class: 'paper-title', text: `${item.rank}. ${item.title}` }),
          el('span', { class: 'badge', text: `score ${item.score}` })),
        el('div', { class: 'paper-meta', text: `${item.published || ''} · ${item.arxiv_id} · ${(item.categories || []).join(', ')}` }),
        el('p', { text: item.why_relevant || item.reason || '' }),
        el('p', { class: 'hint', text: item.reason || '' }),
        repro.has_code
          ? el('p', { class: 'hint' },
            el('span', { text: `代码：${repro.source} · ${repro.confidence}` }),
            el('a', { href: repro.url, target: '_blank', rel: 'noopener', text: ' 仓库链接' }))
          : el('p', { class: 'hint', text: repro.note || '摘要中未发现代码链接' }),
        el('div', { class: 'row wrap card-actions' },
          el('a', { class: 'ghost small', href: item.url, target: '_blank', rel: 'noopener', text: 'arXiv' }),
          el('button', {
            type: 'button', class: 'ghost small', text: '下载到文献库',
            onclick: async (event) => {
              event.target.disabled = true;
              try {
                await api('/api/papers/download', { method: 'POST', body: JSON.stringify({ arxiv_id: item.arxiv_id }) });
                toast('已加入下载队列');
                loadPapers();
              } catch (error) { event.target.disabled = false; toast(error.message, true); }
            },
          }),
          el('button', {
            type: 'button', class: 'ghost small', text: '👍 喜欢',
            onclick: (event) => sendFeedback(item.arxiv_id, 'like', event.target),
          }),
          el('button', {
            type: 'button', class: 'ghost small', text: '👎 不喜欢',
            onclick: (event) => sendFeedback(item.arxiv_id, 'dislike', event.target),
          })));
      host.append(card);
    }
  }

  async function sendFeedback(arxivId, action, btn) {
    btn.disabled = true;
    try {
      await api('/api/recommend/feedback', {
        method: 'POST', body: JSON.stringify({ arxiv_id: arxivId, action }),
      });
      toast(action === 'like' ? '已记录偏好：喜欢' : '已记录偏好：不喜欢');
    } catch (error) {
      btn.disabled = false;
      toast(error.message, true);
    }
  }

  /* ══════════════════════ 写作助手 ══════════════════════ */

  async function writeFramework() {
    const idea = $('#writeIdea').value.trim();
    if (!idea) { toast('请输入研究想法', true); return; }
    const out = $('#writeOutput');
    const btn = $('#writeFramework');
    btn.disabled = true;
    out.replaceChildren(emptyNote('生成中…', '只在文献库真实条目中引用。'));
    try {
      const data = await api('/api/write/framework', {
        method: 'POST',
        body: JSON.stringify({ idea, paper_ids: $('#writeUseAll').checked ? [] : selectedPaperIds() }),
      });
      out.replaceChildren(
        el('pre', { class: 'markdown-text', text: data.markdown || '' }),
        el('h4', { text: '参考文献' }),
        ...(data.references || []).map((r) => el('p', { class: 'hint', text: `${r.title}${r.year ? ` (${r.year})` : ''}` })),
      );
      if ((data.issues || []).length) {
        out.append(el('h4', { text: '校验问题' }),
          ...data.issues.map((i) => el('p', { class: 'warning', text: typeof i === 'string' ? i : JSON.stringify(i) })));
      }
      if ((data.rejected || []).length) {
        out.append(el('h4', { text: '未采用的参考文献' }),
          ...data.rejected.map((r) => el('p', { class: 'hint', text: typeof r === 'string' ? r : JSON.stringify(r) })));
      }
    } catch (error) {
      out.replaceChildren(el('p', { class: 'warning', text: error.message }));
    } finally {
      btn.disabled = false;
    }
  }

  async function makeDiagram() {
    const description = $('#diagramDesc').value.trim();
    if (!description) { toast('请输入结构描述', true); return; }
    const out = $('#diagramOut');
    const issues = $('#diagramIssues');
    try {
      const data = await api('/api/write/diagram', {
        method: 'POST',
        body: JSON.stringify({
          description, format: $('#diagramFormat').value, title: $('#diagramTitle').value.trim(),
        }),
      });
      out.hidden = false;
      out.textContent = data.script || '';
      issues.hidden = !(data.issues || []).length;
      issues.textContent = (data.issues || []).join('；');
    } catch (error) {
      out.hidden = false;
      out.textContent = '';
      issues.hidden = false;
      issues.textContent = error.message;
    }
  }

  async function makeChart() {
    if (!S.chartFile) { toast('请选择 CSV 文件', true); return; }
    const form = new FormData();
    form.append('file', S.chartFile);
    form.append('title', $('#chartTitle').value.trim());
    form.append('figure_no', String(Number($('#chartNo').value) || 1));
    const host = $('#chartOut');
    host.replaceChildren(emptyNote('生成中…', '识别列类型并生成 Matplotlib 脚本。'));
    try {
      const data = await api('/api/write/chart', { method: 'POST', body: form });
      host.replaceChildren();
      if (data.image_url) host.append(el('img', { src: data.image_url, alt: '生成的学术图表' }));
      else host.append(el('p', { class: 'warning', text: '脚本已生成，但本地渲染失败；可在下方脚本中排查依赖。' }));
      host.append(
        el('p', { class: 'hint', text: `图表类型：${data.chart?.name || '—'}（${data.chart?.type || '—'}）` }),
        el('p', { text: data.caption?.zh || '' }),
        el('p', { class: 'hint', text: data.caption?.en || '' }),
      );
      for (const issue of data.issues || []) host.append(el('p', { class: 'warning', text: issue }));
      host.append(el('pre', { class: 'code-out', text: data.script || '' }));
    } catch (error) {
      host.replaceChildren(el('p', { class: 'warning', text: error.message }));
    }
  }

  /* ══════════════════════ 绑定 ══════════════════════ */

  function bind() {
    $('#libUpload').addEventListener('change', (event) => {
      uploadPapers(Array.from(event.target.files || []));
      event.target.value = '';
    });
    $('#libSearch').addEventListener('input', () => loadPapers({ quiet: true }));
    $('#libStatus').addEventListener('change', () => loadPapers());
    $('#libRefresh').addEventListener('click', () => loadPapers());
    $('#libDownload').addEventListener('click', downloadArxiv);
    $('#libArxiv').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); downloadArxiv(); }
    });
    $('#qaForm').addEventListener('submit', askPapers);

    $('#readerPaper').addEventListener('change', (event) => {
      if (!event.target.value) return;
      S.readerId = event.target.value;
      S.reader = null;
      S.translations = {};
      loadReader();
    });
    $('#readerBilingual').addEventListener('change', renderBlocks);
    $('#readerTranslate').addEventListener('click', translateCurrentPage);
    $('#readerTranslateAll').addEventListener('click', translateAll);
    $('#readerExport').addEventListener('click', exportMarkdown);
    $('#readerText').addEventListener('scroll', onReaderScroll);
    $('#readerBlocks').addEventListener('mouseup', showSelectionButton);
    $('#readerSelectionBtn').addEventListener('click', translateSelection);
    // pointerdown 早于选区的收拢，拖选时不会自己把弹层又关掉
    const hideSelectionPop = (event) => {
      const pop = $('#readerSelection');
      if (!pop.hidden && !(event && pop.contains(event.target))) pop.hidden = true;
    };
    document.addEventListener('pointerdown', hideSelectionPop, true);
    // 点进 PDF iframe 时父文档收不到点击，只能靠失焦兜底
    window.addEventListener('blur', () => hideSelectionPop());

    $('#graphBuild').addEventListener('click', buildGraph);
    $('#graphSurvey').addEventListener('click', () => graphAction('survey'));
    $('#graphFuture').addEventListener('click', () => graphAction('future-work'));
    $('#recRefresh').addEventListener('click', () => loadRecommend({ refresh: true }));
    $('#writeFramework').addEventListener('click', writeFramework);
    $('#diagramBtn').addEventListener('click', makeDiagram);
    $('#chartFile').addEventListener('change', (event) => {
      S.chartFile = (event.target.files || [])[0] || null;
    });
    $('#chartBtn').addEventListener('click', makeChart);
  }

  bind();
  RA.onView('library', () => loadPapers({ quiet: true }));
  RA.onView('recommend', () => loadRecommend());
  RA.onView('graph', () => { if (!S.papers.length) loadPapers({ quiet: true }); else fillPaperSelectors(); });
  RA.onView('writing', () => { if (!S.papers.length) loadPapers({ quiet: true }); else fillPaperSelectors(); });

  // 供文献库卡片/问答中的按钮复用
  window.RA = Object.assign(RA, { openReader, loadPapers });

  // app.js 的 init() 在本脚本之前执行，首次进入的视图钩子需要在这里补一次
  const startView = RA.view();
  if (startView === 'library' || startView === 'graph' || startView === 'writing') {
    loadPapers({ quiet: true });
  }
  if (startView === 'recommend') loadRecommend();
  const deepPaper = new URLSearchParams(location.search).get('paper');
  if (deepPaper) openReader(deepPaper);
})();
