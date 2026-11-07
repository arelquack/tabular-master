import React, { useEffect, useRef } from 'react';
import katex from 'katex';

interface MathViewProps {
  math: string;
  block?: boolean;
  className?: string;
}

export const MathView: React.FC<MathViewProps> = ({ math, block = false, className = '' }) => {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        katex.render(math, containerRef.current, {
          displayMode: block,
          throwOnError: false,
        });
      } catch (err) {
        if (containerRef.current) {
          containerRef.current.innerText = math;
        }
      }
    }
  }, [math, block]);

  return (
    <span
      ref={containerRef}
      className={`${block ? 'block my-2.5 overflow-x-auto text-zinc-900 py-1 font-serif select-all' : 'inline-block px-1 font-serif select-all'} ${className}`}
    />
  );
};
