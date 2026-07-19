/**
 * Shows two extra visualizations returned by the backend after each prediction:
 *  - "What the model sees": the 28x28 centered/cropped image actually fed to the CNN
 *  - "Where it focused" (Grad-CAM): a heatmap showing which pixels most
 *    influenced the prediction — red/bright areas mattered most
 */
export default function PredictionVisuals({ preprocessedImage, gradcamImage }) {
  if (!preprocessedImage || !gradcamImage) return null;

  return (
    <div style={{ display: "flex", gap: 24, justifyContent: "center", flexWrap: "wrap" }}>
      <div style={{ textAlign: "center" }}>
        <img
          src={`data:image/png;base64,${preprocessedImage}`}
          alt="Preprocessed 28x28 input fed to the model"
          style={{ border: "1px solid #333", borderRadius: 6, imageRendering: "pixelated" }}
          width={120}
          height={120}
        />
        <p style={{ fontSize: 12, color: "#888", margin: "6px 0 0" }}>What the model sees</p>
      </div>

      <div style={{ textAlign: "center" }}>
        <img
          src={`data:image/png;base64,${gradcamImage}`}
          alt="Grad-CAM heatmap showing which pixels influenced the prediction"
          style={{ border: "1px solid #333", borderRadius: 6, imageRendering: "pixelated" }}
          width={120}
          height={120}
        />
        <p style={{ fontSize: 12, color: "#888", margin: "6px 0 0" }}>Where it focused (Grad-CAM)</p>
      </div>
    </div>
  );
}
