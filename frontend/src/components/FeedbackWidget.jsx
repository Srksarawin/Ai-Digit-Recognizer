import { useState } from "react";

const FEEDBACK_URL = "http://localhost:8000/feedback";

/**
 * After a prediction, asks "was this correct?". If not, lets the user
 * pick the true digit and sends it to the backend so it can be stored
 * for future retraining.
 */
export default function FeedbackWidget({ blob, prediction }) {
  const [status, setStatus] = useState("idle"); // idle | correcting | sent

  if (prediction === null || !blob) return null;

  const sendFeedback = async (trueDigit) => {
    const formData = new FormData();
    formData.append("file", blob, "digit.png");
    formData.append("predicted_digit", prediction);
    formData.append("true_digit", trueDigit);

    try {
      await fetch(FEEDBACK_URL, { method: "POST", body: formData });
      setStatus("sent");
    } catch {
      setStatus("idle");
    }
  };

  if (status === "sent") {
    return <p style={{ fontSize: 13, color: "#4ade80" }}>Thanks — saved for retraining.</p>;
  }

  if (status === "correcting") {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "center" }}>
        <p style={{ fontSize: 13, color: "#aaa" }}>What was the actual digit?</p>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", justifyContent: "center", maxWidth: 280 }}>
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => (
            <button
              key={digit}
              onClick={() => sendFeedback(digit)}
              style={{ padding: "4px 10px", minWidth: 32 }}
            >
              {digit}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
      <span style={{ fontSize: 13, color: "#aaa" }}>Was this correct?</span>
      <button onClick={() => sendFeedback(prediction)} style={{ background: "#166534" }}>
        Yes
      </button>
      <button onClick={() => setStatus("correcting")} style={{ background: "#7f1d1d" }}>
        No
      </button>
    </div>
  );
}
