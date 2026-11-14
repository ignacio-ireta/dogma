/**
 * Original path drawing: the presenter in an oversized coral jacket, mid-stride, pointing at
 * the idea. Limbs are round-capped strokes, so they read as soft tubes; the face is a few
 * marks on a plain head (brow, eye, nose bump, smile, blush).
 */
export function GuideFigure() {
  return (
    <g>
      {/* back arm, behind the jacket */}
      <path className="art-limb art-s-coral" strokeWidth={25} d="M34 96C16 112 8 132 12 154" />
      <circle className="art-skin" cx="13" cy="165" r="11" />
      {/* legs: a long stride */}
      <path className="art-cobalt" d="M40 158C30 200 12 238-8 276L14 286C36 252 60 216 82 178Z" />
      <path
        className="art-cobalt"
        d="M60 160C64 200 78 242 88 284L112 280C110 240 108 198 106 158Z"
      />
      <path className="art-line" d="M70 196 82 178" />
      <path
        className="art-ink"
        d="M-9 272 16 283C15 294 8 298-2 298L-30 296C-32 285-22 276-9 272Z"
      />
      <path
        className="art-ink"
        d="M86 280 113 277C128 284 138 291 138 298L88 300C84 293 84 286 86 280Z"
      />
      {/* neck */}
      <path className="art-limb art-s-skin" strokeWidth={15} d="M70 52 69 70" />
      {/* oversized jacket */}
      <path
        className="art-coral"
        d="M40 68C20 72 10 92 10 122 10 150 18 170 28 178 56 190 92 188 114 172 118 140 118 104 108 80 98 64 60 62 40 68Z"
      />
      <path className="art-line art-stitch" d="M52 70 66 96 82 68" />
      <path className="art-white" d="M36 118 58 122 56 138 34 134Z" />
      <path className="art-line art-stitch" d="M30 160Q66 174 108 158" />
      {/* pointing arm */}
      <path
        className="art-limb art-s-coral"
        strokeWidth={23}
        d="M98 92C116 92 130 80 142 64L158 46"
      />
      <path className="art-line art-stitch" d="M98 106C112 104 122 98 130 90" />
      <circle className="art-skin" cx="164" cy="39" r="10" />
      <path className="art-limb art-s-skin" strokeWidth={6.5} d="M168 34 181 17" />
      <path className="art-limb art-s-skin" strokeWidth={6} d="M157 35 152 31" />
      {/* head */}
      <ellipse className="art-skin" cx="73" cy="33" rx="20" ry="23" transform="rotate(8 73 33)" />
      <circle className="art-skin" cx="57" cy="36" r="5" />
      <path
        className="art-ink"
        d="M52 36C42 12 60-2 80 4 96 4 102 16 96 27 86 16 70 17 64 26 62 31 62 36 60 41 56 41 53 39 52 36Z"
      />
      <circle className="art-ink" cx="50" cy="12" r="10" />
      <circle className="art-coral art-blush" cx="82" cy="40" r="4.5" />
      <circle className="art-ink" cx="82" cy="30" r="1.9" />
      <path className="art-skin" d="M91 27Q100 35 91 40Z" />
      <path
        className="art-line art-face"
        d="M76 24Q81 21 86 24M77 45Q83 50 89 45M58 34Q56 36 58 38"
      />
    </g>
  );
}
