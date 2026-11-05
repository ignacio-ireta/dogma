export function Loading({ what }: { readonly what: string }) {
  return (
    <p className="muted" role="status">
      Loading {what}…
    </p>
  );
}
