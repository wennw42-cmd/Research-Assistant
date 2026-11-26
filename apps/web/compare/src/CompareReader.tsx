import { useState } from 'react';
import { useSyncScroll } from './useSyncScroll';

export interface Pair {
  /** 左右两栏共用的段落 id，必须成对且各栏内唯一 */
  id: string;
  en: string;
  zh: string;
}

export interface CompareReaderProps {
  pairs: Pair[];
  title?: string;
}

/**
 * 中英对照阅读示例组件。
 * 直接抓住任意一栏滚动，另一栏会跟着走；主导侧会实时显示。
 */
export function CompareReader({ pairs, title = '中英对照阅读' }: CompareReaderProps) {
  const [syncEnabled, setSyncEnabled] = useState(true);
  const { leftRef, rightRef, leader } = useSyncScroll({
    enabled: syncEnabled,
    revision: pairs,
  });

  return (
    <div className="reader">
      <header className="reader-bar">
        <strong className="reader-title" title={title}>
          {title}
        </strong>
        <span className="reader-leader" aria-live="polite">
          {syncEnabled ? `主导侧：${leader === 'left' ? '原文' : '译文'}` : '同步已关闭'}
        </span>
        <button
          type="button"
          className="reader-toggle"
          aria-pressed={syncEnabled}
          onClick={() => setSyncEnabled((v) => !v)}
        >
          {syncEnabled ? '关闭同步' : '开启同步'}
        </button>
      </header>

      <main className="panes">
        <div className="pane" ref={leftRef} tabIndex={0} role="group" aria-label="英文原文">
          <div className="pane-inner">
            {pairs.map((p, i) => (
              <article className="para" key={p.id} data-pid={p.id}>
                <span className="para-no">{i + 1}</span>
                {p.en}
              </article>
            ))}
          </div>
        </div>

        <div className="pane" ref={rightRef} tabIndex={0} role="group" aria-label="中文译文">
          <div className="pane-inner">
            {pairs.map((p, i) => (
              <article className="para" key={p.id} data-pid={p.id}>
                <span className="para-no">{i + 1}</span>
                {p.zh}
              </article>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
