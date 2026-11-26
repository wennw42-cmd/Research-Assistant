import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { createScrollSync } from './sync-core';
import type { ScrollSync, Side } from './sync-core';

export interface UseSyncScrollOptions {
  /** 关闭后两栏各自滚动，不再互相同步 */
  enabled?: boolean;
  /** 阅读线在视口高度的比例，默认 0.12 */
  readingLine?: number;
  /** 停手后重新测量并对齐的延迟，默认 120ms */
  idleMs?: number;
  /** 小于该像素差就不写入，默认 1px */
  minDelta?: number;
  /** 段落数据变化时传入（通常直接传 pairs），触发重新测量 */
  revision?: unknown;
}

export interface UseSyncScrollResult {
  /** 绑定到左侧滚动容器 */
  leftRef: RefObject<HTMLDivElement | null>;
  /** 绑定到右侧滚动容器 */
  rightRef: RefObject<HTMLDivElement | null>;
  /** 当前主导侧 */
  leader: Side;
  /** 布局被外部改动后手动重测、重新对齐 */
  remeasure: () => void;
}

/**
 * 中英对照同步滚动 Hook。
 *
 * 使用约定：
 *   - 左右两个滚动容器各包一层内层容器，成对段落都带相同的 data-pid；
 *   - 数据变化时通过 revision 传入新的 pairs 数组。
 */
export function useSyncScroll(
  options: UseSyncScrollOptions = {},
): UseSyncScrollResult {
  const { enabled = true, readingLine, idleMs, minDelta, revision } = options;

  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const ctrlRef = useRef<ScrollSync | null>(null);
  const [leader, setLeader] = useState<Side>('left');

  useEffect(() => {
    void revision; // 仅用于触发重建，值本身不参与逻辑
    const left = leftRef.current;
    const right = rightRef.current;
    if (!enabled || !left || !right) return;

    const ctrl = createScrollSync(left, right, {
      readingLine,
      idleMs,
      minDelta,
      onLeaderChange: setLeader,
    });
    ctrlRef.current = ctrl;
    ctrl.start();
    setLeader(ctrl.getLeader());

    return () => {
      ctrl.stop();
      if (ctrlRef.current === ctrl) ctrlRef.current = null;
    };
  }, [enabled, readingLine, idleMs, minDelta, revision]);

  const remeasure = useCallback(() => {
    ctrlRef.current?.remeasure();
  }, []);

  return { leftRef, rightRef, leader, remeasure };
}
