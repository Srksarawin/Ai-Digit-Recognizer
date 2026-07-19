import { useEffect, useState } from "react";

/**
 * A simplified "input pixels -> hidden nodes -> output nodes" diagram.
 * Node brightness is driven by REAL activation values from the backend
 * (downsampled input pixels, pooled fc1 activations, softmax probabilities)
 * — not decorative. Edge opacity is a simplified visual encoding (average of
 * the two connected nodes' activation), not the actual learned weights,
 * since showing literal weights for every connection wouldn't be readable.
 *
 * Re-animates every time new data arrives via the `vizKey` prop, which
 * should change (e.g. increment) on each new prediction.
 */
export default function NeuralNetVisual({ vizInput, vizHidden, vizOutput, prediction, vizKey }) {
  const [mountKey, setMountKey] = useState(0);

  // Force a remount so CSS fade-in animations restart on every new prediction
  useEffect(() => {
    setMountKey((k) => k + 1);
  }, [vizKey]);

  if (!vizInput || !vizHidden || !vizOutput) return null;

  const width = 640;
  const height = 460;
  const topMargin = 30;
  const usableHeight = height - topMargin * 2;

  const inputX = 70;
  const hiddenX = 330;
  const outputX = 570;

  const inputY = (i) => topMargin + (usableHeight * i) / (vizInput.length - 1);
  const hiddenY = (i) => topMargin + (usableHeight * i) / (vizHidden.length - 1);
  const outputY = (i) => topMargin + (usableHeight * i) / (vizOutput.length - 1);

  // Stagger delays create a left-to-right "signal flowing through" feel
  const inputDelay = (i) => i * 4;
  const hiddenDelay = (i) => 220 + i * 12;
  const outputDelay = (i) => 480 + i * 20;

  const inputColor = (v) => `rgba(255,255,255,${0.15 + v * 0.85})`;
  const hiddenColor = (v) => `rgba(96, 165, 250, ${0.2 + v * 0.8})`; // blue
  const outputColor = (v, isPred) =>
    isPred ? `rgba(74, 222, 128, ${0.4 + v * 0.6})` : `rgba(148, 163, 184, ${0.2 + v * 0.5})`;

  return (
    <div style={{ width: "100%", maxWidth: 640, margin: "0 auto" }}>
      <svg key={mountKey} width="100%" viewBox={`0 0 ${width} ${height}`}>
        <style>{`
          .nn-node, .nn-edge {
            opacity: 0;
            animation: nnFadeIn 0.5s ease forwards;
          }
          @keyframes nnFadeIn {
            from { opacity: 0; }
            to { opacity: var(--final-opacity, 1); }
          }
        `}</style>

        {/* Input -> Hidden edges */}
        {vizInput.map((iv, i) =>
          vizHidden.map((hv, h) => {
            const strength = iv * hv;
            const opacity = 0.03 + strength * 0.5;
            return (
              <line
                key={`ih-${i}-${h}`}
                className="nn-edge"
                x1={inputX}
                y1={inputY(i)}
                x2={hiddenX}
                y2={hiddenY(h)}
                stroke="#93c5fd"
                strokeWidth={0.5}
                style={{ "--final-opacity": opacity, animationDelay: `${(inputDelay(i) + hiddenDelay(h)) / 2}ms` }}
              />
            );
          })
        )}

        {/* Hidden -> Output edges */}
        {vizHidden.map((hv, h) =>
          vizOutput.map((ov, o) => {
            const strength = hv * ov;
            const opacity = 0.05 + strength * 0.6;
            return (
              <line
                key={`ho-${h}-${o}`}
                className="nn-edge"
                x1={hiddenX}
                y1={hiddenY(h)}
                x2={outputX}
                y2={outputY(o)}
                stroke="#86efac"
                strokeWidth={0.6}
                style={{ "--final-opacity": opacity, animationDelay: `${(hiddenDelay(h) + outputDelay(o)) / 2}ms` }}
              />
            );
          })
        )}

        {/* Input nodes */}
        {vizInput.map((v, i) => (
          <circle
            key={`in-${i}`}
            className="nn-node"
            cx={inputX}
            cy={inputY(i)}
            r={4}
            fill={inputColor(v)}
            stroke="#555"
            strokeWidth={0.5}
            style={{ "--final-opacity": 1, animationDelay: `${inputDelay(i)}ms` }}
          />
        ))}

        {/* Hidden nodes */}
        {vizHidden.map((v, i) => (
          <circle
            key={`hid-${i}`}
            className="nn-node"
            cx={hiddenX}
            cy={hiddenY(i)}
            r={9}
            fill={hiddenColor(v)}
            stroke="#3b82f6"
            strokeWidth={0.5}
            style={{ "--final-opacity": 1, animationDelay: `${hiddenDelay(i)}ms` }}
          />
        ))}

        {/* Output nodes + digit labels */}
        {vizOutput.map((v, i) => (
          <g
            key={`out-${i}`}
            className="nn-node"
            style={{ "--final-opacity": 1, animationDelay: `${outputDelay(i)}ms` }}
          >
            <circle
              cx={outputX}
              cy={outputY(i)}
              r={13}
              fill={outputColor(v, i === prediction)}
              stroke={i === prediction ? "#4ade80" : "#64748b"}
              strokeWidth={i === prediction ? 1.5 : 0.5}
            />
            <text
              x={outputX}
              y={outputY(i)}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={11}
              fill="#0f0f0f"
              fontWeight={i === prediction ? "bold" : "normal"}
            >
              {i}
            </text>
            <text
              x={outputX + 26}
              y={outputY(i)}
              textAnchor="start"
              dominantBaseline="central"
              fontSize={10}
              fill="#888"
            >
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* Column labels */}
        <text x={inputX} y={16} textAnchor="middle" fontSize={11} fill="#888">
          Input pixels
        </text>
        <text x={hiddenX} y={16} textAnchor="middle" fontSize={11} fill="#888">
          Hidden layer
        </text>
        <text x={outputX} y={16} textAnchor="middle" fontSize={11} fill="#888">
          Output (0-9)
        </text>
      </svg>
      <p style={{ fontSize: 11, color: "#666", textAlign: "center", marginTop: 4 }}>
        Node brightness reflects real activation values from the model. Edge brightness is a simplified visual cue, not the literal learned weights.
      </p>
    </div>
  );
}
