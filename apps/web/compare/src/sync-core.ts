/**
 * 中英对照同步滚动内核：不依赖 React，任何前端项目都可以直接复用。
 *
 * 三段式算法
 * ──────────
 * 1. 锚点  两栏中成对的段落带同一个 data-pid。测量每段在滚动内容里的
 *          “槽位” [top, top + height)：height 取到下一段起点的距离，段间距
 *          也算进本段，这样阅读线跨过段间距时映射仍然连续。
 * 2. 映射  取主导栏阅读线（默认视口顶部 12%）所在的段落 id 和段内进度 t，
 *          在从属栏找到同 id 段落，落到 top + t * height。全程用绝对位置
 *          换算、不累加增量，所以两栏段落高度不同也不会产生累计漂移。
 * 3. 防回环 程序写入从属栏后记下 echo 时间与目标值；从属栏随后冒出的
 *          scroll 事件若同时命中两者就丢弃，主导侧不变。只有真实输入
 *          （wheel / pointerdown / touchstart / keydown，或未被标记的 scroll）
 *          才会切换主导侧。
 *
 * 布局变化（字体加载、图片撑高、窗口缩放）由 ResizeObserver 与停手后的
 * idle 重测兜底：重测后按同一公式再对齐一次，误差不会跨段累积。
 */

export type Side = 'left' | 'right';

export interface Anchor {
  /** 与另一栏成对的段落 id */
  id: string;
  /** 槽位起点，相对滚动内容 */
  top: number;
  /** 槽位高度：到下一段起点的距离，含段间距 */
  height: number;
}

export interface SyncOptions {
  /** 阅读线在视口中的位置比例，0 = 顶部，默认 0.12 */
  readingLine?: number;
  /** 用户停手多久后重新测量并对齐（毫秒），默认 120 */
  idleMs?: number;
  /** 目标位置与当前值之差小于该像素就不写入，默认 1 */
  minDelta?: number;
  /** 主导侧变化时回调 */
  onLeaderChange?: (side: Side) => void;
}

export interface ScrollSync {
  start(): void;
  stop(): void;
  remeasure(): void;
  syncFrom(side: Side): void;
  getLeader(): Side;
}

const SIDES: readonly Side[] = ['left', 'right'];

/** 这些事件说明用户正在直接操作某一栏，比 scroll 事件更早、更可靠 */
const INTENT_EVENTS = ['wheel', 'touchstart', 'pointerdown', 'keydown'] as const;

/** 程序写入后，从属栏回传的 scroll 事件在此时长内视为回声 */
const ECHO_MS = 150;

const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

const other = (side: Side): Side => (side === 'left' ? 'right' : 'left');

/** 二分查找阅读线命中的段落，返回段内进度 t ∈ [0, 1] */
export function locate(
  anchors: readonly Anchor[],
  y: number,
): { index: number; t: number } {
  if (anchors.length === 0) return { index: -1, t: 0 };
  if (y <= anchors[0].top) return { index: 0, t: 0 };

  let lo = 0;
  let hi = anchors.length - 1;
  let idx = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (anchors[mid].top <= y) {
      idx = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  const a = anchors[idx];
  const h = a.height > 0 ? a.height : 1;
  return { index: idx, t: clamp((y - a.top) / h, 0, 1) };
}

/**
 * 把源栏阅读线 y 映射成目标栏的 scrollTop。
 * 找不到同 id 段落时退回到相同序号的段落。
 */
export function mapTarget(
  src: readonly Anchor[],
  dst: readonly Anchor[],
  dstById: ReadonlyMap<string, Anchor>,
  y: number,
  dstLine: number,
  dstMax: number,
): number | null {
  const { index, t } = locate(src, y);
  if (index < 0) return null;
  const a = src[index];
  const b = dstById.get(a.id) ?? dst[Math.min(index, dst.length - 1)];
  if (!b) return null;
  return clamp(b.top + t * b.height - dstLine, 0, dstMax);
}

export function createScrollSync(
  left: HTMLElement,
  right: HTMLElement,
  options: SyncOptions = {},
): ScrollSync {
  const panes: Record<Side, HTMLElement> = { left, right };
  const readingLine = options.readingLine ?? 0.12;
  const idleMs = options.idleMs ?? 120;
  const minDelta = options.minDelta ?? 1;

  let anchors: Record<Side, Anchor[]> = { left: [], right: [] };
  let byId: Record<Side, Map<string, Anchor>> = {
    left: new Map(),
    right: new Map(),
  };

  let lead: Side = 'left';
  const echoAt: Record<Side, number> = { left: 0, right: 0 };
  const echoVal: Record<Side, number> = { left: NaN, right: NaN };

  let raf = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let running = false;
  let ro: ResizeObserver | null = null;
  const offs: Array<() => void> = [];
  const savedBehavior: Record<Side, string> = {
    left: left.style.scrollBehavior,
    right: right.style.scrollBehavior,
  };

  const maxScroll = (el: HTMLElement): number =>
    Math.max(0, el.scrollHeight - el.clientHeight);

  const measurePane = (el: HTMLElement): Anchor[] => {
    // 滚动内容坐标原点：边框盒顶部 + 上边框 - 当前 scrollTop
    const origin =
      el.getBoundingClientRect().top + el.clientTop - el.scrollTop;
    const raw: Array<{ id: string; top: number; height: number }> = [];

    for (const node of Array.from(
      el.querySelectorAll<HTMLElement>('[data-pid]'),
    )) {
      const id = node.dataset.pid;
      if (!id) continue;
      const rect = node.getBoundingClientRect();
      if (rect.height <= 0) continue;
      raw.push({ id, top: rect.top - origin, height: rect.height });
    }

    return raw.map((cur, i) => {
      const next = raw[i + 1];
      return {
        id: cur.id,
        top: cur.top,
        height: next ? Math.max(1, next.top - cur.top) : Math.max(1, cur.height),
      };
    });
  };

  const rebuild = (): void => {
    for (const side of SIDES) {
      const list = measurePane(panes[side]);
      anchors[side] = list;
      const map = new Map<string, Anchor>();
      for (const a of list) if (!map.has(a.id)) map.set(a.id, a);
      byId[side] = map;
    }
  };

  const apply = (from: Side): void => {
    if (!running) return;
    const src = panes[from];
    const dstSide = other(from);
    const dst = panes[dstSide];
    if (anchors[from].length === 0 || anchors[dstSide].length === 0) return;

    const srcLine = src.clientHeight * readingLine;
    const dstLine = dst.clientHeight * readingLine;
    const target = mapTarget(
      anchors[from],
      anchors[dstSide],
      byId[dstSide],
      src.scrollTop + srcLine,
      dstLine,
      maxScroll(dst),
    );
    if (target === null) return;
    if (Math.abs(dst.scrollTop - target) < minDelta) return;

    echoAt[dstSide] = performance.now() + ECHO_MS;
    echoVal[dstSide] = target;
    dst.scrollTop = target;
  };

  const flush = (): void => {
    raf = 0;
    apply(lead);
  };

  const schedule = (): void => {
    if (!raf) raf = requestAnimationFrame(flush);
  };

  const bumpIdle = (): void => {
    if (timer !== undefined) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = undefined;
      rebuild();
      apply(lead);
    }, idleMs);
  };

  const setLead = (side: Side): void => {
    if (lead === side) return;
    lead = side;
    options.onLeaderChange?.(side);
  };

  const onScroll = (side: Side): void => {
    const el = panes[side];
    const isEcho =
      performance.now() < echoAt[side] &&
      Math.abs(el.scrollTop - echoVal[side]) <= 2;
    if (isEcho) return;
    setLead(side);
    schedule();
    bumpIdle();
  };

  const onIntent = (side: Side): void => setLead(side);

  const listen = (el: HTMLElement, type: string, fn: EventListener): void => {
    el.addEventListener(type, fn, { passive: true });
    offs.push(() => el.removeEventListener(type, fn));
  };

  const watch = (): void => {
    if (typeof ResizeObserver === 'undefined') return;
    ro = new ResizeObserver(() => {
      rebuild();
      apply(lead);
    });
    for (const side of SIDES) {
      ro.observe(panes[side]);
      // 约定：段落列表放在一个内层容器里，容器高度变化即可代表内容变化
      const inner = panes[side].firstElementChild;
      if (inner) ro.observe(inner);
    }
  };

  const watchFonts = (): void => {
    const doc = panes.left.ownerDocument;
    if (!doc.fonts) return;
    doc.fonts.ready
      .then(() => {
        if (!running) return;
        rebuild();
        apply(lead);
      })
      .catch(() => {});
  };

  const start = (): void => {
    if (running) return;
    running = true;
    for (const side of SIDES) {
      const el = panes[side];
      el.style.scrollBehavior = 'auto'; // 程序写入不能被 CSS smooth 拖成动画
      listen(el, 'scroll', () => onScroll(side));
      for (const type of INTENT_EVENTS) listen(el, type, () => onIntent(side));
    }
    rebuild();
    watch();
    watchFonts();
    schedule();
  };

  const stop = (): void => {
    if (!running) return;
    running = false;
    for (const off of offs.splice(0)) off();
    if (raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
    if (timer !== undefined) {
      clearTimeout(timer);
      timer = undefined;
    }
    ro?.disconnect();
    ro = null;
    for (const side of SIDES) {
      panes[side].style.scrollBehavior = savedBehavior[side];
    }
  };

  const remeasure = (): void => {
    rebuild();
    apply(lead);
  };

  return {
    start,
    stop,
    remeasure,
    syncFrom: apply,
    getLeader: () => lead,
  };
}
