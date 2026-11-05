const paths = {
  today: "M8 3v4m8-4v4M4 10h16M5 5h14a1 1 0 0 1 1 1v14H4V6a1 1 0 0 1 1-1Zm3 10h3m-3 3h6",
  problem: "m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 16",
  map: "m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2V5Zm6-2v16m6-14v16",
  review: "M4 7h12a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7Zm4-4h12v13M8 12h7m-7 4h4",
  progress: "M4 4v16h17M8 15l4-5 4 2 5-7M17 5h4v4",
  arrow: "M5 12h14m-5-5 5 5-5 5",
  sun: "M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
  leaf: "M5 19c-4-9 3-14 15-15 0 12-5 19-14 15m-2 2L15 10",
  mark: "M5 18V6h6a6 6 0 0 1 0 12H5Zm5-7h9m-3-3 3 3-3 3",
  mock: "M12 7v5l3 2m6-2a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  welcome: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm3.5-12.5-2 5-5 2 2-5 5-2Z",
} as const;

export type IconName = keyof typeof paths;

export function Icon({ name, className = "" }: { name: IconName; className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={`icon ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={paths[name]} />
    </svg>
  );
}
