/* Research Assistant 工作台 · 零构建前端
 * 所有动态内容一律走 textContent / DOM 构造，不拼 innerHTML（模型输出不可信）。
 */
(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const VIEW_META = {
    library: ['文献库', 'PDF 上传与解析 · 版本去重 · 单篇/多篇问答'],
    reader: ['阅读器', '版面保留 · 目录 · 引用跳转 · 划词翻译 · 双语对照'],
    graph: ['引用图谱', '库内引用建图 · 基石/边缘挖掘 · 综述与 Future Work'],
    recommend: ['推荐', 'arXiv 增量 · 相关性与代码可用性 · 反馈学习'],
    writing: ['写作助手', '想法 → 框架 · 架构图 · 数据图表'],
    translate: ['学术翻译', '中译英 · 英译中 · 公式与引用原样保留'],
    general: ['通用大模型', '模型问答 · 图片理解 · 文生图 · 语音识别与朗读'],
    chat: ['对话 / 任务', 'Supervisor 路由 · DAG 规划 · 工具调用 · 证据校验'],
    status: ['系统状态', '后端、模型与 OCR 配置一眼看清'],
  };

  // research.js 注册的视图钩子：进入视图时刷新数据
  const viewHooks = {};

  const STAT_LABELS = {
    paragraphs: '段落',
    chunks: '片段',
    terms: '术语',
    source_chars: '原文字符',
    translation_chars: '译文字符',
    protected_spans: '保护片段',
  };

  const ATTACH_KIND = { pdf: 'PDF', docx: 'Word', image: '图片', text: '文本' };
  const ATTACH_LIMIT = 12000;
  const ATTACH_MAX_BYTES = 25 * 1024 * 1024;
  const GENERAL_IMAGE_LIMIT = 12000;
  const DEMO_ZH =
    '本文提出一种面向长文档的稀疏注意力框架[1]。设序列长度为 $n$，键维度为 $d_k$，' +
    '则标准自注意力的计算复杂度为 $O(n^2 d_k)$。已有研究（Zhang et al., 2020）表明，' +
    '全局注意力在长序列上会带来显著的内存开销。我们在 WMT 2014 英德翻译任务上设置 ' +
    '`d_model = 512`，并采用 $h = 8$ 个注意力头。实验表明，该框架在保持翻译质量的同时' +
    '显著降低了显存占用（见表 1）；然而，当文档长度超过 4096 个词元时，稀疏模式的选择' +
    '仍会引入额外误差，因此需要进一步研究自适应稀疏度调度。';
  const DEMO_EN =
    'The Transformer replaces recurrence with self-attention [1]. For a sequence of length $n$, ' +
    'the computational complexity of self-attention is $O(n^2 d)$, where $d$ is the model dimension. ' +
    'In our implementation we set `d_model = 512` and use $h = 8$ attention heads. ' +
    'Although the model achieves strong results on the WMT 2014 English-to-German translation task, ' +
    'its quadratic memory footprint becomes prohibitive for long documents, and prior work has ' +
    'therefore explored sparse attention patterns [2, 3].';

  const state = {
    threadId: '',
    busy: false,
    translation: null,
    status: null,
    attachment: null,
    uploading: false,
    uploadToken: 0,
    // 通用大模型模块：与 Agent 的线程/附件状态完全隔离
    translateDirection: 'zh2en',
    generalThreadId: '',
    generalMessages: [],
    generalImage: null,
    generalImageMode: 'vision',
    generalUploading: false,
    generalUploadToken: 0,
    generalBusy: false,
    generalGenerating: false,
    generalRecognizing: false,
    generalRecognition: null,
  };

  /* ── 基础工具 ───────────────────────────────────── */
  let toastTimer = 0;
  function toast(message, isError = false) {
    const el = $('#toast');
    el.textContent = message;
    el.classList.toggle('error', isError);
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 2800);
  }

  async function api(path, options = {}) {
    const isForm = options.body instanceof FormData;
    const res = await fetch(path, {
      ...options,
      headers: {
        ...(isForm ? {} : { 'Content-Type': 'application/json' }),
        ...(options.headers || {}),
      },
    });
    const raw = await res.text();
    let data = null;
    try { data = raw ? JSON.parse(raw) : null; } catch { data = { detail: raw }; }
    if (!res.ok) {
      const detail = data && data.detail;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail || `HTTP ${res.status}`));
    }
    return data;
  }

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'hidden') node.hidden = Boolean(value);
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
      else node.setAttribute(key, value);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function fmtValue(value) {
    if (value == null) return '—';
    if (typeof value === 'number') {
      return Number.isInteger(value) ? String(value) : String(Math.round(value * 1000) / 1000);
    }
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  function stamp() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  }

  function download(filename, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = el('a', { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ── 视图切换 ───────────────────────────────────── */
  function setView(name) {
    if (!VIEW_META[name]) name = 'translate';
    $$('.nav-item').forEach((b) => {
      const active = b.dataset.view === name;
      b.classList.toggle('active', active);
      b.setAttribute('aria-current', active ? 'page' : 'false');
    });
    $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
    $('#viewTitle').textContent = VIEW_META[name][0];
    $('#viewSubtitle').textContent = VIEW_META[name][1];
    if (name !== 'general') {
      stopRecognition();
      stopSpeaking();
    }
    if (viewHooks[name]) {
      try { viewHooks[name](); } catch (error) { toast(error.message || '视图加载失败', true); }
    }
    if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
  }

  // 供 research.js 复用同一套请求/DOM 工具与视图切换
  window.RA = {
    api, el, toast, download, setView,
    view: () => (location.hash.replace('#', '') || 'translate'),
    onView: (name, fn) => { viewHooks[name] = fn; },
  };

  /* ── 状态 / 健康检查 ─────────────────────────────── */
  function setHealth(ok, detail) {
    const dot = $('#healthDot');
    dot.className = `dot ${ok ? 'ok' : 'bad'}`;
    $('#healthText').textContent = ok ? '服务在线' : '服务离线';
    $('#healthSub').textContent = detail;
    $('#stHealth').textContent = ok ? '在线' : '离线';
    $('#stService').textContent = detail;
  }

  function renderModel(info) {
    const badge = $('#modelBadge');
    if (!info) {
      badge.textContent = '配置未知';
      badge.className = 'badge bad';
      return;
    }
    const mock = Boolean(info.mock);
    const label = mock ? 'MOCK · 未配置模型' : `${info.provider} · ${info.name}`;
    badge.textContent = label;
    badge.className = `badge ${mock ? 'warn' : ''}`;
    badge.title = info.key_set ? 'API Key 已配置（不显示内容）' : '未检测到 API Key';

    $('#stModel').textContent = info.name || '—';
    $('#stProvider').textContent = `${info.provider || '—'}${info.key_set ? ' · key 已配置' : ' · key 缺失'}`;
    $('#stMock').textContent = mock ? 'Mock 降级' : '真实模型';
    $('#stModeHint').textContent = mock
      ? '不调用外部模型，不伪造译文'
      : 'OpenAI 兼容接口';
  }

  function renderOcr(ocr) {
    const langs = (ocr && ocr.languages) || [];
    $('#stOcr').textContent = ocr && ocr.lang ? ocr.lang : '未配置';
    $('#stTessdata').textContent = ocr && ocr.tessdata
      ? `语言包：${langs.length ? langs.join(', ') : '目录为空'}`
      : '未找到 tessdata';
    $('#stTessdata').title = ocr && ocr.tessdata ? ocr.tessdata : '';
  }

  function renderWarning(info) {
    const warn = $('#statusWarning');
    const messages = [];
    if (info && info.mock) {
      messages.push('当前是 Mock 模式：Agent 可离线运行，但学术翻译与通用问答会拒绝执行（不伪造结果）。在 apps/api/.env 填入 LLM_API_KEY 并重启即可。');
    }
    if (info && !info.key_set) {
      messages.push('未检测到 LLM_API_KEY，OpenAI 兼容模型不可用。');
    }
    warn.textContent = messages.join(' ');
    warn.hidden = messages.length === 0;
  }

  async function refreshStatus() {
    const btn = $('#refreshBtn');
    btn.disabled = true;
    try {
      const [health, status] = await Promise.all([
        api('/health').catch((e) => { throw new Error(e.message); }),
        api('/api/status'),
      ]);
      state.status = status;
      setHealth(health.status === 'ok', health.service || '/health');
      renderModel(status.model);
      renderOcr(status.ocr);
      renderWarning(status.model);
    } catch (err) {
      setHealth(false, err.message || '无法连接后端');
      $('#modelBadge').textContent = '无法读取配置';
      $('#modelBadge').className = 'badge bad';
      renderWarning(null);
      toast('后端状态读取失败：' + err.message, true);
    } finally {
      btn.disabled = false;
    }
  }

  /* ── 学术翻译（中译英 / 英译中） ─────────────────── */
  function parseGlossary(raw, direction = 'zh2en') {
    const en2zh = direction === 'en2zh';
    const text = raw.trim();
    if (!text) return {};
    const out = {};
    if (text.startsWith('[') || text.startsWith('{')) {
      let data;
      try { data = JSON.parse(text); } catch { throw new Error('术语表 JSON 解析失败'); }
      if (Array.isArray(data)) {
        for (const item of data) {
          if (!item || typeof item !== 'object' || !item.zh || !item.en) continue;
          const src = en2zh ? item.en : item.zh;
          const dst = en2zh ? item.zh : item.en;
          out[String(src).trim()] = String(dst).trim();
        }
      } else if (data && typeof data === 'object') {
        for (const [src, dst] of Object.entries(data)) {
          if (src.trim() && String(dst).trim()) out[src.trim()] = String(dst).trim();
        }
      }
      return out;
    }
    for (const line of text.split(/\r?\n/)) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const m = t.split(/\s*(?:=|＝|→)\s*/, 2);
      if (m.length === 2 && m[0].trim() && m[1].trim()) {
        out[m[0].trim()] = m[1].trim();
      } else {
        throw new Error(`术语表行格式不对：${t}（应为 ${en2zh ? 'English=中文' : '中文=English'}）`);
      }
    }
    return out;
  }

  function setTranslateDirection(mode) {
    const next = mode === 'en2zh' ? 'en2zh' : 'zh2en';
    const zh2en = next === 'zh2en';
    state.translateDirection = next;
    $$('.seg[data-tr-mode]').forEach((btn) => {
      const active = btn.dataset.trMode === next;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    $('#sourceTitle').textContent = zh2en ? '中文原文' : '英文原文';
    $('#sourceSub').textContent = zh2en ? '摘要、全文或段落均可' : '英文摘要、论文段落或全文';
    $('#zhText').placeholder = zh2en
      ? '把中文摘要或论文段落粘贴到这里……'
      : '把英文摘要或论文段落粘贴到这里……';
    $('#glossaryHint').innerHTML = zh2en
      ? '每行 <code>中文=English</code>，或直接粘贴 JSON 数组'
      : '每行 <code>English=中文</code>，或直接粘贴 JSON 数组';
    $('#glossaryInput').placeholder = zh2en
      ? '稀疏注意力=sparse attention\n长文档=long document'
      : 'sparse attention=稀疏注意力\nlong document=长文档';
    $('#translateBtn').textContent = zh2en ? '翻译为学术英文' : '翻译为中文';
    $('#translationEmptyHint').textContent = zh2en
      ? '输入中文后点击“翻译为学术英文”。'
      : '输入英文后点击“翻译为中文”。';
    const shown = (state.translation && state.translation.direction) || next;
    $('#outputTitle').textContent = shown === 'en2zh' ? '中文输出' : '英文输出';
  }

  function tabTo(name) {
    $$('.tab').forEach((t) => {
      const active = t.dataset.tab === name;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    $$('.tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === name));
  }

  function renderTranslation(result) {
    const en2zh = (result.direction || state.translateDirection) === 'en2zh';
    const translation = result.translation || '';
    $('#translationEmpty').hidden = Boolean(translation);
    const out = $('#translationOut');
    out.hidden = !translation;
    out.textContent = translation;

    const compare = $('#compareOut');
    compare.replaceChildren();
    const pairs = result.pairs || [];
    if (!pairs.length) {
      compare.append(el('div', { class: 'empty' },
        el('strong', { text: '暂无对照' }),
        el('span', { text: '本次返回没有段落对照。' })));
    } else {
      compare.append(el('div', { class: 'compare-head' },
        el('span', { text: en2zh ? 'English' : '中文' }),
        el('span', { text: en2zh ? '中文' : 'English' })));
      for (const pair of pairs) {
        compare.append(el('div', { class: 'compare-row' },
          el('div', { class: 'compare-cell zh', text: (en2zh ? pair.en : pair.zh) || '' }),
          el('div', { class: 'compare-cell en', text: (en2zh ? pair.zh : pair.en) || '' })));
      }
    }

    const glossary = $('#glossaryOut');
    glossary.replaceChildren();
    const terms = result.glossary || [];
    if (!terms.length) {
      glossary.append(el('div', { class: 'empty' },
        el('strong', { text: en2zh ? '未设置术语' : '未抽取到术语' }),
        el('span', {
          text: en2zh
            ? '可展开高级选项，填写 English=中文 术语表后重试。'
            : '可展开高级选项，手动填写术语表后重试。',
        })));
    } else {
      const tbody = el('tbody');
      for (const term of terms) {
        tbody.append(el('tr', {},
          el('td', { text: en2zh ? term.en || '' : term.zh || '' }),
          el('td', { text: en2zh ? term.zh || '' : term.en || '' }),
          el('td', { text: term.note || '—' })));
      }
      glossary.append(el('table', { class: 'glossary-table' },
        el('thead', {}, el('tr', {},
          el('th', { text: en2zh ? 'English' : '中文' }),
          el('th', { text: en2zh ? '中文' : 'English' }),
          el('th', { text: '说明' }))),
        tbody));
    }

    const issuesEl = $('#issuesOut');
    issuesEl.replaceChildren();
    const issues = result.issues || [];
    if (!issues.length) {
      issuesEl.append(el('div', { class: 'issue-ok', text: '占位符、术语与格式校验通过。' }));
    } else {
      issuesEl.append(el('ul', { class: 'issue-list' },
        issues.map((issue) => el('li', { text: String(issue) }))));
    }

    const statsEl = $('#statsOut');
    statsEl.replaceChildren();
    for (const [key, label] of Object.entries(STAT_LABELS)) {
      const value = (result.stats || {})[key];
      if (value == null) continue;
      statsEl.append(el('span', { class: 'stat' }, label, el('strong', { text: String(value) })));
    }

    const stats = result.stats || {};
    const elapsed = result.latency_ms != null ? ` · ${result.latency_ms} ms` : '';
    $('#translateMeta').textContent =
      `${stats.paragraphs || 0} 段 · ${stats.terms || 0} 个术语 · ` +
      `${stats.source_chars || 0} → ${stats.translation_chars || 0} 字符${elapsed}`;

    for (const id of ['#copyBtn', '#downloadMdBtn', '#downloadJsonBtn']) $(id).disabled = false;
    $('#outputTitle').textContent = en2zh ? '中文输出' : '英文输出';
    $('#translationEmptyHint').textContent = en2zh
      ? '输入英文后点击“翻译为中文”。'
      : '输入中文后点击“翻译为学术英文”。';
    state.translation = result;
    tabTo('translation');
  }

  async function translate() {
    const direction = state.translateDirection;
    const en2zh = direction === 'en2zh';
    const text = $('#zhText').value.trim();
    if (!text) {
      toast(en2zh ? '请先输入英文原文' : '请先输入中文原文', true);
      $('#zhText').focus();
      return;
    }
    let glossary;
    try {
      glossary = parseGlossary($('#glossaryInput').value, direction);
    } catch (err) {
      toast(err.message, true);
      return;
    }
    const maxChars = Number($('#maxChars').value) || 3000;
    const maxTerms = Number($('#maxTerms').value) || 0;

    const btn = $('#translateBtn');
    btn.disabled = true;
    btn.textContent = '翻译中…';
    $('#translateMeta').textContent = '正在调用模型，长文可能需要几十秒…';
    try {
      const result = await api('/api/translate', {
        method: 'POST',
        body: JSON.stringify({
          text,
          direction,
          max_chars: maxChars,
          max_terms: maxTerms,
          glossary,
        }),
      });
      renderTranslation(result);
      toast(`翻译完成 · ${result.glossary.length} 个术语`);
    } catch (err) {
      $('#translateMeta').textContent = '翻译失败';
      toast('翻译失败：' + err.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = en2zh ? '翻译为中文' : '翻译为学术英文';
    }
  }

  function loadDemo() {
    $('#zhText').value = state.translateDirection === 'en2zh' ? DEMO_EN : DEMO_ZH;
    $('#zhText').focus();
    toast('已载入示例');
  }

  /* ── 通用大模型（独立模块，不走 Agent） ───────────── */
  function generalEmpty() {
    return el('div', { class: 'empty chat-empty' },
      el('strong', { text: '通用问答模块' }),
      el('span', { text: '直接提问；上传图片后可做图片理解或提取文字；语音可用麦克风输入、点“朗读”听回答。' }));
  }

  function appendGeneralMessage(role, text) {
    const log = $('#generalLog');
    log.querySelector('.chat-empty')?.remove();
    const bubble = el('div', { class: 'bubble', text });
    const meta = el('div', { class: 'msg-meta' });
    const wrap = el('div', { class: `msg ${role}` },
      el('span', { class: 'msg-role', text: role === 'user' ? '你' : '模型' }),
      bubble,
      meta);
    log.append(wrap);
    log.scrollTop = log.scrollHeight;
    return { wrap, bubble, meta };
  }

  function generalFinish(message, answer, model) {
    message.bubble.style.color = '';
    message.bubble.textContent = answer;
    if (model) message.meta.append(chip(String(model), 'muted'));
    message.meta.append(el('button', {
      type: 'button', class: 'ghost small speak-btn', text: '朗读',
      onclick: () => speak(answer),
    }));
    $('#generalLog').scrollTop = $('#generalLog').scrollHeight;
  }

  function renderGeneralImage() {
    const chipEl = $('#generalImageChip');
    const item = state.generalImage;
    if (!item) {
      chipEl.hidden = true;
      chipEl.classList.remove('uploading');
      chipEl.title = '';
      return;
    }
    chipEl.hidden = false;
    chipEl.classList.toggle('uploading', Boolean(item.uploading) || state.generalUploading);
    $('#generalImageName').textContent = item.filename || '';
    const modeEl = $('#generalImageMode');
    if (modeEl) modeEl.value = item.mode || state.generalImageMode;
    if (item.uploading) {
      $('#generalImageMeta').textContent = '正在识别文字…';
      return;
    }
    if ((item.mode || state.generalImageMode) === 'vision') {
      $('#generalImageMeta').textContent = '图片理解 · 发送时分析';
      chipEl.title = '';
      return;
    }
    const bits = [];
    if (item.chars != null) bits.push(`${item.chars} 字`);
    if (item.truncated) bits.push('已截断');
    $('#generalImageMeta').textContent = bits.join(' · ') || '未识别到文字';
    chipEl.title = (item.warnings || []).join('；');
  }

  function clearGeneralImage() {
    state.generalUploadToken += 1;
    state.generalImage = null;
    state.generalUploading = false;
    const btn = $('#generalImageBtn');
    if (btn) btn.disabled = false;
    renderGeneralImage();
  }

  async function runGeneralImageOcr(file, token) {
    const form = new FormData();
    form.append('file', file);
    try {
      const data = await api('/api/general/image', { method: 'POST', body: form });
      if (token !== state.generalUploadToken) return;
      state.generalImage = { ...data, file, mode: 'ocr' };
      toast(`图片文字提取完成 · ${data.chars} 字`);
    } catch (err) {
      if (token !== state.generalUploadToken) return;
      state.generalImage = null;
      toast('图片文字提取失败：' + err.message, true);
    } finally {
      if (token === state.generalUploadToken) {
        state.generalUploading = false;
        $('#generalImageBtn').disabled = false;
        renderGeneralImage();
      }
    }
  }

  async function handleGeneralImage(file) {
    if (!file) return;
    if (!/^image\//.test(file.type) && !/\.(png|jpe?g|tiff?|bmp|webp)$/i.test(file.name)) {
      toast('请选择 PNG / JPG / TIFF / BMP / WEBP 图片', true);
      return;
    }
    if (file.size > ATTACH_MAX_BYTES) {
      toast('图片超过 25 MB，请压缩后上传', true);
      return;
    }
    const mode = state.generalImageMode === 'ocr' ? 'ocr' : 'vision';
    if (mode === 'vision' && file.size > 10 * 1024 * 1024) {
      toast('图片理解上限 10 MB，请压缩后上传，或切换为“提取文字”', true);
      return;
    }
    const token = ++state.generalUploadToken;
    state.generalImage = { filename: file.name, file, mode, uploading: mode === 'ocr' };
    state.generalUploading = mode === 'ocr';
    $('#generalImageBtn').disabled = state.generalUploading;
    renderGeneralImage();
    if (mode === 'ocr') {
      await runGeneralImageOcr(file, token);
    } else {
      toast('图片已添加，发送时做图片理解');
    }
  }

  function handleGeneralImageMode() {
    const mode = $('#generalImageMode').value === 'ocr' ? 'ocr' : 'vision';
    state.generalImageMode = mode;
    const item = state.generalImage;
    if (!item) return;
    item.mode = mode;
    if (mode === 'vision') {
      state.generalUploadToken += 1;
      state.generalUploading = false;
      item.uploading = false;
      $('#generalImageBtn').disabled = false;
      renderGeneralImage();
      return;
    }
    if (!item.text && item.file) {
      const token = ++state.generalUploadToken;
      item.uploading = true;
      state.generalUploading = true;
      $('#generalImageBtn').disabled = true;
      renderGeneralImage();
      runGeneralImageOcr(item.file, token);
    } else {
      renderGeneralImage();
    }
  }

  function generalHistoryForApi() {
    const out = [];
    let total = 0;
    for (let i = state.generalMessages.length - 1; i >= 0; i -= 1) {
      const message = state.generalMessages[i];
      if (out.length >= 20 || total + message.content.length > 50000) break;
      out.unshift(message);
      total += message.content.length;
    }
    return out;
  }

  async function sendGeneral(event) {
    event?.preventDefault();
    if (state.generalBusy) return;
    if (state.generalUploading) {
      toast('图片还在处理中，请稍候', true);
      return;
    }
    if (state.generalRecognizing) stopRecognition();
    const input = $('#generalInput');
    let query = input.value.trim();
    const image = state.generalImage;
    const useVision = Boolean(
      image && image.file && (image.mode || state.generalImageMode) === 'vision'
    );
    if (!query && !image) return;
    if (!query) {
      query = useVision
        ? '请详细描述这张图片的内容，并提取其中的文字。'
        : '请阅读图片中的文字并说明内容。';
    }
    const ocr = !useVision && image
      ? ((image.text || '').trim() || '（图片中没有识别到文字。）').slice(0, GENERAL_IMAGE_LIMIT)
      : '';
    const payload = !useVision && image ? `${query}\n\n【图片 OCR 文字】\n${ocr}` : query;

    state.generalMessages.push({ role: 'user', content: payload });
    input.value = '';
    const userMessage = appendGeneralMessage('user', query);
    if (image) {
      userMessage.meta.append(chip(`图片 ${image.filename}`, 'muted'));
      userMessage.meta.append(chip(useVision ? '图片理解' : 'OCR', 'muted'));
      if (!useVision) addDetails(userMessage.meta, '图片识别文字', ocr);
    }

    state.generalBusy = true;
    const btn = $('#generalSendBtn');
    btn.disabled = true;
    btn.textContent = useVision ? '识别中…' : '生成中…';
    const pending = appendGeneralMessage('assistant', useVision ? '正在理解图片…' : '正在生成…');
    pending.bubble.style.color = '#8e867a';

    try {
      let result;
      if (useVision) {
        const form = new FormData();
        form.append('file', image.file);
        form.append('question', query);
        form.append('thread_id', state.generalThreadId);
        result = await api('/api/general/image/chat', { method: 'POST', body: form });
      } else {
        result = await api('/api/general/chat', {
          method: 'POST',
          body: JSON.stringify({
            messages: generalHistoryForApi(),
            thread_id: state.generalThreadId,
          }),
        });
      }
      const answer = result.answer || '（模型没有返回内容。）';
      generalFinish(pending, answer, result.model);
      state.generalMessages.push({ role: 'assistant', content: answer });
      if (image) clearGeneralImage();
    } catch (err) {
      pending.bubble.style.color = '';
      pending.bubble.textContent = '请求失败：' + err.message;
      pending.bubble.classList.add('error');
      const last = state.generalMessages[state.generalMessages.length - 1];
      if (last && last.role === 'user') state.generalMessages.pop();
      toast((useVision ? '图片理解失败：' : '通用问答失败：') + err.message, true);
    } finally {
      state.generalBusy = false;
      btn.disabled = false;
      btn.textContent = '发送';
      $('#generalLog').scrollTop = $('#generalLog').scrollHeight;
      input.focus();
    }
  }

  function setGeneralMode(mode) {
    const next = mode === 'image' ? 'image' : 'chat';
    $$('.seg[data-gen-mode]').forEach((btn) => {
      const active = btn.dataset.genMode === next;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    const chatPane = $('#generalChatPane');
    const imagePane = $('#generalImagePane');
    if (chatPane) chatPane.hidden = next !== 'chat';
    if (imagePane) imagePane.hidden = next !== 'image';
    if (next === 'image') {
      stopRecognition();
      $('#genPrompt')?.focus();
    } else {
      $('#generalInput')?.focus();
    }
  }

  function resetGeneratedImages() {
    const box = $('#genResult');
    if (!box) return;
    box.replaceChildren();
    box.hidden = true;
  }

  async function generateGeneralImage() {
    if (state.generalGenerating) return;
    const prompt = $('#genPrompt').value.trim();
    if (!prompt) {
      toast('请输入图片提示词', true);
      $('#genPrompt').focus();
      return;
    }
    const size = $('#genSize').value;
    const n = Number($('#genCount').value) || 1;
    const negative = $('#genNegative').value.trim();
    state.generalGenerating = true;
    const btn = $('#genImageBtn');
    const box = $('#genResult');
    btn.disabled = true;
    btn.textContent = '生成中…';
    box.hidden = false;
    box.replaceChildren(el('div', { class: 'empty' },
      el('strong', { text: '正在生成图片' }),
      el('span', { text: `通常需要 10–30 秒，请稍候…（${n} 张）` })));
    try {
      const data = await api('/api/general/image/generate', {
        method: 'POST',
        body: JSON.stringify({ prompt, size, n, negative_prompt: negative }),
      });
      renderGeneratedImages(data);
      toast(`已生成 ${data.count || 0} 张图片`);
    } catch (err) {
      box.replaceChildren(el('div', { class: 'empty' },
        el('strong', { text: '生成失败' }),
        el('span', { text: err.message })));
      toast('图片生成失败：' + err.message, true);
    } finally {
      state.generalGenerating = false;
      btn.disabled = false;
      btn.textContent = '生成图片';
    }
  }

  function renderGeneratedImages(data) {
    const box = $('#genResult');
    const images = (data && data.images) || [];
    box.hidden = false;
    if (!images.length) {
      box.replaceChildren(el('div', { class: 'empty' },
        el('strong', { text: '没有返回图片' }),
        el('span', { text: '请调整提示词后重试。' })));
      return;
    }
    const cards = images.map((image, index) => {
      const src = image.local_url || image.url || '';
      return el('div', { class: 'gen-card' },
        src
          ? el('img', { src, alt: data.prompt || '生成图片', loading: 'lazy' })
          : el('div', { class: 'empty', text: '图片下载失败' }),
        el('div', { class: 'gen-meta' },
          chip(data.model || '通义万相', 'muted'),
          src ? el('a', {
            class: 'ghost small',
            href: src,
            target: '_blank',
            rel: 'noopener',
            download: `generated-${index + 1}.png`,
            text: '下载',
          }) : null));
    });
    const head = el('p', { class: 'hint' },
      `实际提示词：${data.actual_prompt || data.prompt || ''}`);
    box.replaceChildren(head, el('div', { class: 'gen-grid' }, ...cards));
    if (data.warnings && data.warnings.length) {
      box.append(el('p', { class: 'hint', text: data.warnings.join('；') }));
    }
  }

  function newGeneralThread() {
    if (state.generalBusy || state.generalGenerating) {
      toast('当前任务还在进行，稍后再试', true);
      return;
    }
    stopRecognition();
    stopSpeaking();
    state.generalMessages = [];
    state.generalThreadId = `gen-${Date.now().toString(36)}`;
    saveGeneralThread();
    clearGeneralImage();
    $('#generalLog').replaceChildren(generalEmpty());
    $('#genPrompt').value = '';
    resetGeneratedImages();
    toast('已新建通用对话');
    $('#generalInput').focus();
  }

  /* ── 语音识别 / 朗读（浏览器原生） ───────────────── */
  function speechRecognitionCtor() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }

  function stopRecognition() {
    try { state.generalRecognition?.stop(); } catch { /* 已结束 */ }
    state.generalRecognizing = false;
    const btn = $('#generalMicBtn');
    if (btn) {
      btn.classList.remove('recording');
      btn.textContent = '语音';
      btn.setAttribute('aria-pressed', 'false');
    }
  }

  function toggleRecognition() {
    const Ctor = speechRecognitionCtor();
    if (!Ctor) {
      toast('当前浏览器不支持语音识别，请使用 Chrome / Edge', true);
      return;
    }
    if (state.generalRecognizing) {
      stopRecognition();
      return;
    }
    const recognition = new Ctor();
    state.generalRecognition = recognition;
    recognition.lang = $('#genLang').value || 'zh-CN';
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onstart = () => {
      state.generalRecognizing = true;
      const btn = $('#generalMicBtn');
      btn.classList.add('recording');
      btn.textContent = '停止';
      btn.setAttribute('aria-pressed', 'true');
    };
    recognition.onresult = (event) => {
      let text = '';
      for (const result of event.results) text += result[0].transcript;
      text = text.trim();
      if (text) {
        const input = $('#generalInput');
        input.value = input.value ? `${input.value} ${text}` : text;
        input.focus();
      }
    };
    recognition.onerror = (event) => {
      const labels = {
        'not-allowed': '麦克风未授权',
        'no-speech': '没有听到语音',
        'audio-capture': '没有可用麦克风',
        network: '语音服务网络错误',
      };
      toast('语音识别失败：' + (labels[event.error] || event.error), true);
    };
    recognition.onend = () => {
      state.generalRecognizing = false;
      const btn = $('#generalMicBtn');
      btn.classList.remove('recording');
      btn.textContent = '语音';
      btn.setAttribute('aria-pressed', 'false');
    };
    try {
      recognition.start();
    } catch (err) {
      toast('无法启动语音识别：' + err.message, true);
    }
  }

  function plainForSpeech(text) {
    return String(text || '')
      .replace(/\[\[ev:[^\]]+\]\]/g, '')
      .replace(/[#*_`>]/g, '')
      .replace(/\n{2,}/g, '\n')
      .trim();
  }

  function stopSpeaking() {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  }

  function speak(text) {
    const tts = window.speechSynthesis;
    if (!tts) {
      toast('当前浏览器不支持语音朗读', true);
      return;
    }
    const spoken = plainForSpeech(text);
    if (!spoken) return;
    tts.cancel();
    const utterance = new SpeechSynthesisUtterance(spoken.slice(0, 8000));
    const selected = $('#genLang').value || 'zh-CN';
    utterance.lang = /[一-鿿]/.test(spoken) ? 'zh-CN' : selected;
    const voice = (tts.getVoices() || []).find((v) => {
      const lang = String(v.lang || '').replace('_', '-');
      return lang === utterance.lang || lang.startsWith(utterance.lang.slice(0, 2));
    });
    if (voice) utterance.voice = voice;
    utterance.rate = 1;
    utterance.onerror = (event) => {
      if (!['interrupted', 'canceled'].includes(event.error)) {
        toast('朗读失败：' + event.error, true);
      }
    };
    tts.speak(utterance);
  }

  /* ── 对话 ───────────────────────────────────────── */
  function saveThread() {
    try { localStorage.setItem('ra.thread_id', state.threadId); } catch { /* 隐私模式忽略 */ }
  }

  function initThread() {
    let id = '';
    try { id = localStorage.getItem('ra.thread_id') || ''; } catch { /* ignore */ }
    if (!id) id = `web-${Date.now().toString(36)}`;
    state.threadId = id;
    saveThread();
    $('#threadLabel').textContent = id;
  }

  function saveGeneralThread() {
    try { localStorage.setItem('ra.general_thread_id', state.generalThreadId); } catch { /* 隐私模式忽略 */ }
  }

  function initGeneralThread() {
    let id = '';
    try { id = localStorage.getItem('ra.general_thread_id') || ''; } catch { /* ignore */ }
    if (!id) id = `gen-${Date.now().toString(36)}`;
    state.generalThreadId = id;
    saveGeneralThread();
  }

  function chatEmpty() {
    return el('div', { class: 'empty chat-empty' },
      el('strong', { text: '开始一轮科研任务' }),
      el('span', { text: '输入问题，或上传 PDF、Word、图片、文本作为附件。' }));
  }

  function attachLabel(item) {
    return ATTACH_KIND[item && item.kind] || '附件';
  }

  function renderAttachment() {
    const chipEl = $('#attachChip');
    const item = state.attachment;
    if (!item) {
      chipEl.hidden = true;
      chipEl.classList.remove('uploading');
      chipEl.title = '';
      return;
    }
    chipEl.hidden = false;
    chipEl.classList.toggle('uploading', Boolean(item.uploading) || state.uploading);
    $('#attachKind').textContent = item.uploading ? '解析中' : attachLabel(item);
    $('#attachName').textContent = item.filename || '';
    if (item.uploading) {
      $('#attachMeta').textContent = '正在解析，请稍候…';
      return;
    }
    const bits = [];
    if (item.chars != null) bits.push(`${item.chars} 字`);
    const pages = item.meta && item.meta.pages;
    if (pages) bits.push(`${pages} 页`);
    const scanned = item.meta && item.meta.scanned_pages;
    if (scanned) bits.push(`${scanned} 页 OCR`);
    if (item.truncated) bits.push('已截断');
    $('#attachMeta').textContent = bits.join(' · ');
    chipEl.title = (item.warnings || []).join('；');
  }

  function clearAttachment() {
    state.uploadToken += 1;
    state.attachment = null;
    state.uploading = false;
    $('#attachBtn').disabled = false;
    renderAttachment();
  }

  async function handleAttachment(file) {
    if (!file) return;
    if (file.size > ATTACH_MAX_BYTES) {
      toast('附件超过 25 MB，请拆分后上传', true);
      return;
    }
    const token = ++state.uploadToken;
    state.uploading = true;
    state.attachment = { filename: file.name, kind: '', uploading: true };
    $('#attachBtn').disabled = true;
    renderAttachment();
    const form = new FormData();
    form.append('file', file);
    try {
      const data = await api('/api/parse', { method: 'POST', body: form });
      if (token !== state.uploadToken) return;
      state.uploading = false;
      state.attachment = data;
      renderAttachment();
      toast(`已解析 ${data.filename} · ${data.chars} 字`);
    } catch (err) {
      if (token !== state.uploadToken) return;
      state.uploading = false;
      state.attachment = null;
      renderAttachment();
      toast('附件解析失败：' + err.message, true);
    } finally {
      if (token === state.uploadToken) {
        state.uploading = false;
        $('#attachBtn').disabled = false;
        renderAttachment();
      }
    }
  }

  function appendMessage(role, text) {
    const log = $('#chatLog');
    log.querySelector('.chat-empty')?.remove();
    const bubble = el('div', { class: 'bubble', text });
    const wrap = el('div', { class: `msg ${role}` },
      el('span', { class: 'msg-role', text: role === 'user' ? '你' : '助手' }),
      bubble,
      el('div', { class: 'msg-meta' }));
    log.append(wrap);
    log.scrollTop = log.scrollHeight;
    return { wrap, bubble, meta: wrap.querySelector('.msg-meta') };
  }

  function chip(text, kind = '') {
    return el('span', { class: `chip ${kind}`, text });
  }

  function addDetails(meta, title, value) {
    if (value == null || (Array.isArray(value) && !value.length) ||
        (typeof value === 'object' && !Array.isArray(value) && !Object.keys(value).length)) {
      return;
    }
    const body = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    meta.append(el('details', { class: 'details' },
      el('summary', { text: title }),
      el('pre', { text: body })));
  }

  function renderAgentResult(result, message) {
    const { bubble, meta } = message;
    if (result.status === 'awaiting_human') {
      bubble.textContent = '计划已生成，等待人工确认后才会执行。';
      meta.append(chip('等待确认', 'warn'));
      if (result.graph_thread) meta.append(chip(`线程 ${result.graph_thread.split('#t').pop()}`, 'muted'));
      addHitl(message, result.graph_thread);
      return;
    }
    const answer = result.answer || '（本轮没有生成文本回答，展开下方详情查看任务结果。）';
    bubble.textContent = answer;
    if (result.route) meta.append(chip(String(result.route)));
    if (result.intent) meta.append(chip(String(result.intent), 'muted'));
    if (result.complexity) meta.append(chip(`复杂度 ${fmtValue(result.complexity)}`, 'muted'));
    if (typeof result.confidence === 'number') {
      const percent = result.confidence <= 1 ? Math.round(result.confidence * 100) + '%' : String(result.confidence);
      meta.append(chip(`置信度 ${percent}`, 'muted'));
    }
    if (result.latency_ms != null) meta.append(chip(`${result.latency_ms} ms`, 'muted'));
    if (result.model) meta.append(chip(String(result.model), 'muted'));

    addDetails(meta, `计划 (${(result.plan || []).length})`, result.plan);
    addDetails(meta, '任务结果', result.task_results);
    addDetails(meta, '校验', result.verify);
    addDetails(meta, `证据 (${(result.evidence || []).length})`, result.evidence);
    addDetails(meta, `轨迹 (${(result.trace || []).length})`, result.trace);
  }

  function addHitl(message, graphThread) {
    if (!graphThread) return;
    const row = el('div', { class: 'hitl' },
      el('button', {
        type: 'button', class: 'primary', text: '批准执行',
        onclick: () => resumeHitl(graphThread, 'approved', message),
      }),
      el('button', {
        type: 'button', class: 'ghost', text: '驳回',
        onclick: () => resumeHitl(graphThread, 'rejected', message),
      }));
    message.wrap.append(row);
    message.wrap.scrollIntoView({ block: 'nearest' });
  }

  async function resumeHitl(graphThread, decision, message) {
    message.wrap.querySelector('.hitl')?.remove();
    const busy = appendMessage('assistant', decision === 'approved' ? '已批准，继续执行…' : '已驳回，正在调整计划…');
    try {
      const result = await api('/resume', {
        method: 'POST',
        body: JSON.stringify({
          graph_thread: graphThread,
          decision,
          feedback: decision === 'rejected' ? '用户驳回了该计划，请调整方案后重新执行。' : '',
        }),
      });
      renderAgentResult(result, busy);
      toast(decision === 'approved' ? '已恢复执行' : '已提交驳回意见');
    } catch (err) {
      busy.bubble.textContent = '恢复执行失败：' + err.message;
      busy.bubble.classList.add('error');
      toast('恢复失败：' + err.message, true);
    }
  }

  async function sendChat(event) {
    event?.preventDefault();
    if (state.busy) return;
    if (state.uploading) {
      toast('附件还在解析中，请稍候', true);
      return;
    }
    const input = $('#chatInput');
    let query = input.value.trim();
    const hasAttachment = Boolean(state.attachment);
    if (!query && hasAttachment) query = '请阅读附件并总结主要内容。';
    if (!query) return;
    const attachmentText = hasAttachment
      ? ((state.attachment.text || '').trim() || '（附件中没有提取到可用文字。）').slice(0, ATTACH_LIMIT)
      : '';

    state.busy = true;
    $('#chatSendBtn').disabled = true;
    $('#chatSendBtn').textContent = '处理中…';
    input.value = '';
    const userMessage = appendMessage('user', query);
    if (hasAttachment) userMessage.meta.append(chip(`附件 ${state.attachment.filename}`, 'muted'));
    const pending = appendMessage('assistant', '正在规划与检索…');
    pending.bubble.style.color = '#8e867a';

    try {
      const result = await api('/chat', {
        method: 'POST',
        body: JSON.stringify({
          query,
          thread_id: state.threadId,
          user_id: 'web',
          attachment: attachmentText,
        }),
      });
      if (result.status === 'error') throw new Error(result.error || 'Agent 执行失败');
      pending.bubble.style.color = '';
      renderAgentResult(result, pending);
    } catch (err) {
      pending.bubble.style.color = '';
      pending.bubble.textContent = '请求失败：' + err.message;
      pending.bubble.classList.add('error');
      toast('对话失败：' + err.message, true);
    } finally {
      state.busy = false;
      $('#chatSendBtn').disabled = false;
      $('#chatSendBtn').textContent = '发送';
      $('#chatLog').scrollTop = $('#chatLog').scrollHeight;
      input.focus();
    }
  }

  async function newThread() {
    if (state.busy) {
      toast('当前任务还没结束，稍后再试', true);
      return;
    }
    if (!window.confirm('新建会话会清空该线程的后端记忆，确定继续？')) return;
    const old = state.threadId;
    try {
      await api(`/thread/${encodeURIComponent(old)}`, { method: 'DELETE' });
    } catch (err) {
      toast('旧线程清理失败（不影响新会话）：' + err.message, true);
    }
    state.threadId = `web-${Date.now().toString(36)}`;
    saveThread();
    $('#threadLabel').textContent = state.threadId;
    const log = $('#chatLog');
    log.replaceChildren(chatEmpty());
    toast('已新建会话');
    $('#chatInput').focus();
  }

  /* ── 事件绑定 ───────────────────────────────────── */
  function bind() {
    $$('.nav-item').forEach((btn) => btn.addEventListener('click', () => setView(btn.dataset.view)));
    $('#refreshBtn').addEventListener('click', refreshStatus);

    $('#demoBtn').addEventListener('click', loadDemo);
    $('#clearBtn').addEventListener('click', () => {
      $('#zhText').value = '';
      $('#zhText').focus();
    });
    $('#fileInput').addEventListener('change', async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      if (file.size > 2 * 1024 * 1024) {
        toast('文件超过 2 MB，请拆分为更小的文本', true);
        event.target.value = '';
        return;
      }
      try {
        $('#zhText').value = await file.text();
        toast(`已导入 ${file.name}`);
      } catch (err) {
        toast('读取文件失败：' + err.message, true);
      } finally {
        event.target.value = '';
      }
    });
    $('#translateBtn').addEventListener('click', translate);
    $$('.seg[data-tr-mode]').forEach((btn) => {
      btn.addEventListener('click', () => setTranslateDirection(btn.dataset.trMode));
    });
    $('#glossaryInput').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) translate();
    });

    $('#copyBtn').addEventListener('click', async () => {
      const text = state.translation && state.translation.translation;
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        toast('译文已复制');
      } catch {
        toast('复制失败：浏览器未授权剪贴板', true);
      }
    });
    $('#downloadMdBtn').addEventListener('click', () => {
      const result = state.translation;
      if (!result) return;
      download(`translation-${stamp()}.md`, result.markdown || result.translation || '', 'text/markdown;charset=utf-8');
    });
    $('#downloadJsonBtn').addEventListener('click', () => {
      const result = state.translation;
      if (!result) return;
      download(`translation-${stamp()}.json`, JSON.stringify(result, null, 2), 'application/json;charset=utf-8');
    });
    $$('.tab').forEach((tab) => tab.addEventListener('click', () => tabTo(tab.dataset.tab)));

    $('#attachBtn').addEventListener('click', () => $('#attachInput').click());
    $('#attachInput').addEventListener('change', (event) => {
      const file = event.target.files && event.target.files[0];
      event.target.value = '';
      handleAttachment(file);
    });
    $('#attachRemove').addEventListener('click', clearAttachment);

    $('#chatForm').addEventListener('submit', sendChat);
    $('#chatInput').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendChat();
      }
    });
    $('#newThreadBtn').addEventListener('click', newThread);

    $('#generalForm').addEventListener('submit', sendGeneral);
    $('#generalInput').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendGeneral();
      }
    });
    $('#generalImageBtn').addEventListener('click', () => $('#generalImageInput').click());
    $('#generalImageInput').addEventListener('change', (event) => {
      const file = event.target.files && event.target.files[0];
      event.target.value = '';
      handleGeneralImage(file);
    });
    $('#generalImageRemove').addEventListener('click', clearGeneralImage);
    $('#generalMicBtn').addEventListener('click', toggleRecognition);
    $$('.seg[data-gen-mode]').forEach((btn) => {
      btn.addEventListener('click', () => setGeneralMode(btn.dataset.genMode));
    });
    $('#generalImageMode').addEventListener('change', handleGeneralImageMode);
    $('#genImageBtn').addEventListener('click', generateGeneralImage);
    $('#genPrompt').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        generateGeneralImage();
      }
    });
    $('#genNewBtn').addEventListener('click', newGeneralThread);
  }

  /* ── 启动 ───────────────────────────────────────── */
  function init() {
    bind();
    initThread();
    initGeneralThread();
    const hash = location.hash.replace('#', '');
    setView(VIEW_META[hash] ? hash : 'translate');
    window.addEventListener('hashchange', () => {
      const next = location.hash.replace('#', '');
      if (VIEW_META[next]) setView(next);
    });
    refreshStatus();
  }

  init();
})();
