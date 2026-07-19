export default function ConfidenceBars({ probabilities }) {
  if (!probabilities) return null;
  const maxIdx = probabilities.indexOf(Math.max(...probabilities));

  return (
    <div style={{ width: 280 }}>
      {probabilities.map((p, digit) => (
        <div key={digit} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
          <span style={{ width: 12, fontWeight: digit === maxIdx ? "bold" : "normal" }}>{digit}</span>
          <div style={{ background: "#222", flex: 1, height: 14, borderRadius: 4, overflow: "hidden" }}>
            <div
              style={{
                width: `${(p * 100).toFixed(1)}%`,
                background: digit === maxIdx ? "#4ade80" : "#666",
                height: "100%",
                transition: "width 0.2s ease",
              }}
            />
          </div>
          <span style={{ width: 40, fontSize: 12 }}>{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
