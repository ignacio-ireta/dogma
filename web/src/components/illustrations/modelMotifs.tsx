export const modelMotifs = {
  neural: (
    <g>
      <path
        className="art-line art-muted-line"
        d="m30 32 60-13 60 13M30 32l60 28 60-28M30 32l60 69 60-13M30 88l60-69 60 69M30 88l60-28 60 28M30 88l60 13m0-82 60 13m-60 69 60-69"
      />
      <g className="art-coral">
        <circle cx="30" cy="32" r="12" />
        <circle cx="30" cy="88" r="12" />
      </g>
      <g className="art-violet">
        <circle cx="90" cy="19" r="12" />
        <circle cx="90" cy="60" r="12" />
        <circle cx="90" cy="101" r="12" />
      </g>
      <g className="art-cobalt">
        <circle cx="150" cy="32" r="12" />
        <circle cx="150" cy="88" r="12" />
      </g>
    </g>
  ),
  attention: (
    <g>
      <path
        className="art-line art-muted-line"
        d="m29 24 0 70m40-70L29 94m80-70L29 94m120-70L29 94m120-70-40 70m0-70 40 70m-80-70 40 70"
      />
      <path
        className="art-line art-coral-line art-bold-line"
        d="m69 24 40 70m0-70-80 70m120-70L69 94"
      />
      <g className="art-violet">
        <rect x="16" y="10" width="26" height="22" rx="5" />
        <rect x="56" y="10" width="26" height="22" rx="5" />
        <rect x="96" y="10" width="26" height="22" rx="5" />
        <rect x="136" y="10" width="26" height="22" rx="5" />
      </g>
      <g className="art-cobalt">
        <rect x="16" y="88" width="26" height="22" rx="5" />
        <rect x="56" y="88" width="26" height="22" rx="5" />
        <rect x="96" y="88" width="26" height="22" rx="5" />
        <rect x="136" y="88" width="26" height="22" rx="5" />
      </g>
    </g>
  ),
  tokens: (
    <g>
      <path
        className="art-violet-soft"
        d="M20 13h135q12 0 12 12v31q0 12-12 12H72L50 82V68H20Q8 68 8 56V25q0-12 12-12"
      />
      <path className="art-line" d="M27 31h35m11 0h23m11 0h39M27 49h19m11 0h39m11 0h20" />
      <rect className="art-coral" x="17" y="92" width="39" height="19" rx="4" />
      <rect className="art-violet" x="65" y="92" width="26" height="19" rx="4" />
      <rect className="art-cobalt" x="100" y="92" width="62" height="19" rx="4" />
    </g>
  ),
  retrieval: (
    <g>
      <rect className="art-violet-soft" x="19" y="12" width="49" height="66" rx="5" />
      <rect className="art-cobalt-soft" x="10" y="22" width="49" height="66" rx="5" />
      <path className="art-line" d="M21 37h25M21 49h25M21 61h18M21 73h25" />
      <circle className="art-line art-coral-line art-bold-line" cx="88" cy="42" r="17" />
      <path className="art-line art-coral-line art-bold-line" d="m100 55 14 14" />
      <path
        className="art-violet"
        d="M120 71h39q11 0 11 11v15q0 11-11 11h-14l-13 8v-8h-12q-11 0-11-11V82q0-11 11-11"
      />
      <path className="art-line art-white-line" d="M124 85h31m-31 10h20" />
      <path className="art-line" d="M43 100v9h51m-7-7 7 7-7 7" />
    </g>
  ),
};
