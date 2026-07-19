import { useState } from "react";
import DigitCanvas from "./components/DigitCanvas";
import ConfidenceBars from "./components/ConfidenceBars";
import FeedbackWidget from "./components/FeedbackWidget";
import PredictionVisuals from "./components/PredictionVisuals";
import NeuralNetVisual from "./components/NeuralNetVisual";

const BACKEND_URL = "http://localhost:8000/predict";

function App() {
  const [probabilities, setProbabilities] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [lastBlob, setLastBlob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [vizInput, setVizInput] = useState(null);
  const [vizHidden, setVizHidden] = useState(null);
  const [vizKey, setVizKey] = useState(0);

  const handlePredict = async (blob) => {
    if (!blob) {
      setProbabilities(null);
      setPrediction(null);
      setLastBlob(null);
      setError(null);
      setPreprocessedImage(null);
      setGradcamImage(null);
      setVizInput(null);
      setVizHidden(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", blob, "digit.png");

      const res = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Backend returned an error");

      const data = await res.json();
      setPrediction(data.prediction);
      setProbabilities(data.probabilities);
      setPreprocessedImage(data.preprocessed_image);
      setGradcamImage(data.gradcam_image);
      setVizInput(data.viz_input);
      setVizHidden(data.viz_hidden);
      setVizKey((k) => k + 1);
      setLastBlob(blob);
    } catch (err) {
      setError("Could not reach backend — is FastAPI running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 20,
        marginTop: 40,
        paddingBottom: 60,
      }}
    >
      <div style={{ textAlign: "center" }}>
        <h1 style={{ marginBottom: 4 }}>Digit Recognizer</h1>
        <p style={{ color: "#888", fontSize: 14 }}>Draw a digit from 0-9, then click Predict</p>
      </div>

      <DigitCanvas onPredict={handlePredict} />

      {loading && <p>Predicting...</p>}
      {error && <p style={{ color: "#f87171" }}>{error}</p>}

      {prediction !== null && !loading && (
        <>
          <h1 style={{ fontSize: 48, margin: 0 }}>{prediction}</h1>
          <ConfidenceBars probabilities={probabilities} />
          <NeuralNetVisual
            vizInput={vizInput}
            vizHidden={vizHidden}
            vizOutput={probabilities}
            prediction={prediction}
            vizKey={vizKey}
          />
          <PredictionVisuals preprocessedImage={preprocessedImage} gradcamImage={gradcamImage} />
          <FeedbackWidget blob={lastBlob} prediction={prediction} />
        </>
      )}
    </div>
  );
}

export default App;
