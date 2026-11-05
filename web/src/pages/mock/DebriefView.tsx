// Loads one scored mock's debrief and shows it.
import { describeError } from "../../api/client";
import { Loading } from "../../components/Loading";
import { Notice } from "../../components/Notice";
import { useDebrief } from "../../hooks/useMock";
import { DebriefPanel } from "./DebriefPanel";

export function DebriefView({
  sessionId,
  onBack,
}: {
  readonly sessionId: string;
  readonly onBack: () => void;
}) {
  const debrief = useDebrief(sessionId);
  return (
    <div className="mock-debrief">
      {debrief.isPending && <Loading what="your debrief" />}
      {debrief.isError && <Notice tone="error">{describeError(debrief.error)}</Notice>}
      {debrief.data && <DebriefPanel debrief={debrief.data} />}
      <button className="button" type="button" onClick={onBack}>
        Back to mocks
      </button>
    </div>
  );
}
