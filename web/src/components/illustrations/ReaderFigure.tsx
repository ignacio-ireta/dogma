/**
 * Original path drawing: the reader in a violet shirt, perched on a stool, eyes down on an
 * open book. Drawn with the same soft limbs and few-mark face as the guide.
 */
export function ReaderFigure() {
  return (
    <g>
      {/* stool */}
      <rect className="art-violet-soft" x="70" y="131" width="92" height="13" rx="6.5" />
      <path className="art-line" d="M84 144 76 206M146 144 158 206M82 178H152" />
      {/* far leg, then near leg */}
      <path
        className="art-ink"
        d="M150 114C120 108 92 110 74 124 64 138 60 166 56 194L76 197C80 176 86 158 94 148 116 148 140 146 156 140Z"
      />
      <path
        className="art-ink"
        d="M132 116C102 110 72 114 54 128 44 142 40 168 36 194L58 197C62 176 68 158 76 148 98 150 124 146 140 140Z"
      />
      <path className="art-line art-stitch" d="M76 148C70 138 62 132 54 130" />
      <path
        className="art-cobalt"
        d="M30 190 60 195C60 205 56 209 48 209L12 207C10 198 19 192 30 190Z"
      />
      <path
        className="art-cobalt"
        d="M54 190 78 195C78 205 74 209 66 209L60 209C62 200 60 194 54 190Z"
      />
      {/* far arm */}
      <path className="art-limb art-s-violet" strokeWidth={20} d="M104 68C90 82 74 88 56 86" />
      <circle className="art-peach" cx="48" cy="85" r="9" />
      {/* book */}
      <path className="art-coral" d="M-6 62 28 70 50 104 14 98Z" />
      <path className="art-white art-outlined" d="M28 70 54 62 76 94 50 104Z" />
      <path className="art-white art-outlined" d="M-2 66 28 70 50 104 18 99Z" />
      <path
        className="art-line art-muted-line"
        d="M8 76 24 79M13 84 29 87M40 74 52 71M46 82 58 79"
      />
      {/* neck and shirt */}
      <path className="art-limb art-s-peach" strokeWidth={16} d="M108 46 112 64" />
      <path
        className="art-violet"
        d="M98 54C122 44 148 60 154 92 158 112 154 130 148 140 124 148 98 146 82 138 82 112 86 80 98 54Z"
      />
      <path className="art-line art-stitch" d="M100 58 112 74 124 56" />
      {/* near arm */}
      <path className="art-limb art-s-violet" strokeWidth={21} d="M136 74C126 100 100 112 70 106" />
      <path className="art-line art-stitch" d="M122 76C116 92 104 100 90 102" />
      <circle className="art-peach" cx="62" cy="104" r="10" />
      {/* head */}
      <ellipse
        className="art-peach"
        cx="104"
        cy="27"
        rx="19"
        ry="22"
        transform="rotate(-8 104 27)"
      />
      <circle className="art-peach" cx="120" cy="31" r="5" />
      <path
        className="art-ink"
        d="M124 34C136 30 142 16 134 6 128-6 108-8 96 0 86 2 82 10 84 18 94 12 108 12 116 20 118 26 118 32 120 38 122 38 123 36 124 34Z"
      />
      <circle className="art-ink" cx="136" cy="26" r="12" />
      <circle className="art-ink" cx="130" cy="44" r="9" />
      <circle className="art-coral art-blush" cx="95" cy="35" r="4.5" />
      <path className="art-peach" d="M87 24Q77 32 87 37Z" />
      <path
        className="art-line art-face"
        d="M90 27Q95 31 100 27M89 22Q94 19 100 21M90 42Q95 47 101 43M119 29Q121 31 119 33"
      />
    </g>
  );
}
