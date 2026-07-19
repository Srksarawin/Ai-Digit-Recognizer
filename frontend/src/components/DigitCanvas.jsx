import { useRef, useState, useEffect } from "react";

const CANVAS_SIZE = 280;

export default function DigitCanvas({ onPredict, canvasRef: externalRef }) {
  const internalRef = useRef(null);
  const canvasRef = externalRef || internalRef;
  const isDrawing = useRef(false);
  const [hasDrawing, setHasDrawing] = useState(false);

  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 18; // thick stroke, mimics typical MNIST stroke width
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, [canvasRef]);

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const startDraw = (e) => {
    e.preventDefault();
    isDrawing.current = true;
    const { x, y } = getPos(e);
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(x, y);
    setHasDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing.current) return;
    e.preventDefault();
    const { x, y } = getPos(e);
    const ctx = canvasRef.current.getContext("2d");
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDraw = () => {
    isDrawing.current = false;
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setHasDrawing(false);
    onPredict(null);
  };

  const handlePredict = () => {
    canvasRef.current.toBlob((blob) => {
      onPredict(blob);
    }, "image/png");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: "center" }}>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        style={{
          border: "2px solid #444",
          borderRadius: 8,
          touchAction: "none",
          cursor: "crosshair",
        }}
        onMouseDown={startDraw}
        onMouseMove={draw}
        onMouseUp={stopDraw}
        onMouseLeave={stopDraw}
        onTouchStart={startDraw}
        onTouchMove={draw}
        onTouchEnd={stopDraw}
      />
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={handlePredict} disabled={!hasDrawing}>
          Predict
        </button>
        <button onClick={clearCanvas}>Clear</button>
      </div>
    </div>
  );
}
