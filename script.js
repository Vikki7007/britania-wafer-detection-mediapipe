// ===== WAFER DETECTION VARIABLES =====
const URL = "./"; // model.json, metadata.json, weights.bin must be in same folder
let model, maxPredictions;
let waferFrameCount = 0;
const REQUIRED_FRAMES = 2; // how many consecutive frames needed for PASS
let waferDetected = false; // NEW: tracks if wafer has been detected

// ===== EATING DETECTION VARIABLES =====
// Lip landmark indices (468-point Face Mesh)
const LIP_INDICES = [
  // inner upper
  78, 191, 80, 81, 82, 13, 312, 311, 310,
  // inner lower
  178, 88, 95, 402, 318, 324, 308
];

// State to merge results from both models
let latestFace = null;
let latestHands = null;

// Chew detection config/state
const EAT_WINDOW_MS = 8000; // 8s window
const CHEW_TARGET = 1;      // need >=2 chews in window
const OPEN_THR = 0.08;      // mouth open threshold (ratio)
const CLOSE_THR = 0.04;     // mouth close threshold (ratio; lower than OPEN_THR)

let mouthState = "closed";   // "open" | "closed"
let chewEvents = [];         // timestamps (ms) of each close event
let eatingDetected = false;

// Wafer-to-mouth gating (both tips in 10–40 px band for hold time)
const CONTACT_REQUIRED_MS = 100; // hold time (ms)
const TOUCH_MIN_PX = 1;          // inner radius of acceptable band
const TOUCH_MAX_PX = 60;          // outer radius of acceptable band
let waferTaken = false;           // becomes true after hold completes
let contactStartTs = null;        // when both tips first within band
let lastHoldMs = 0;               // progress
let holdingPrev = false;          // for transition logs

// ===== DOM ELEMENTS =====
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Status elements
const waferStatusEl = document.getElementById('wafer-status');
const faceStatusEl = document.getElementById('face-status');
const handStatusEl = document.getElementById('hand-status');
const waferMouthStatusEl = document.getElementById('wafer-mouth-status');
const eatingStatusEl = document.getElementById('eating-status');
const chewCountEl = document.getElementById('chew-count');

// ===== HELPER FUNCTIONS =====
function getLipPoints(landmarks, w, h) {
  return LIP_INDICES.map(i => ({
    x: Math.round(landmarks[i].x * w),
    y: Math.round(landmarks[i].y * h)
  }));
}

function euclideanDistance(p1, p2) {
  const dx = p2.x - p1.x, dy = p2.y - p1.y;
  return Math.hypot(dx, dy);
}

function getCenterPoint(points) {
  let sx = 0, sy = 0;
  for (const p of points) { sx += p.x; sy += p.y; }
  return { x: sx / points.length, y: sy / points.length };
}

function pushChewEvent(ts) {
  chewEvents.push(ts);
  const cutoff = ts - EAT_WINDOW_MS;
  while (chewEvents.length && chewEvents[0] < cutoff) chewEvents.shift();
  const wasEating = eatingDetected;
  eatingDetected = chewEvents.length >= CHEW_TARGET;
  if (!wasEating && eatingDetected) console.log("EATING ✅");
}

function updateStatusElements() {
  // Update chew count
  chewCountEl.textContent = chewEvents.length;
  
  // Update eating status
  if (eatingDetected) {
    eatingStatusEl.textContent = "EATING ✅";
    eatingStatusEl.className = "status-value pass";
  } else {
    eatingStatusEl.textContent = waferDetected ? (waferTaken ? "Ready to detect eating" : "Bring wafer to mouth") : "Waiting for wafer";
    eatingStatusEl.className = "status-value fail";
  }
  
  // Update wafer to mouth status
  if (waferTaken) {
    waferMouthStatusEl.textContent = "WAFER TAKEN TO MOUTH ✅";
    waferMouthStatusEl.className = "status-value pass";
  } else if (waferDetected) {
    waferMouthStatusEl.textContent = "Bring wafer to your mouth";
    waferMouthStatusEl.className = "status-value warning";
  } else {
    waferMouthStatusEl.textContent = "Waiting for wafer detection";
    waferMouthStatusEl.className = "status-value fail";
  }
}

// ===== WAFER DETECTION FUNCTIONS =====
async function initWaferModel() {
  try {
    waferStatusEl.textContent = "Loading model...";
    waferStatusEl.className = "status-value warning";
    
    // Load model + metadata
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";
    model = await tmImage.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();
    
    waferStatusEl.textContent = "Model loaded ✅";
    waferStatusEl.className = "status-value pass";
    console.log("Wafer detection model loaded successfully");
  } catch (err) {
    console.error("Error loading wafer model:", err);
    waferStatusEl.textContent = "❌ Failed to load model";
    waferStatusEl.className = "status-value fail";
  }
}

async function predictWafer() {
  if (!model || waferDetected) return; // Don't run if already detected
  
  try {
    // Use webcam.canvas just like the working code
    const prediction = await model.predict(video);
    
    const waferProb = prediction[0].probability;
    const noWaferProb = prediction[1].probability;

    // Debug: Log probabilities every 30 frames
    if (Math.random() < 0.033) { // ~1 in 30 frames
      console.log(`wafer: ${(waferProb * 100).toFixed(1)}%, no wafer: ${(noWaferProb * 100).toFixed(1)}%`);
    }

    // Use the exact same logic as your working code
    if (waferProb > noWaferProb && waferProb > 0.8) {
      waferFrameCount++;
    } else {
      waferFrameCount = 0; // reset if any frame is not wafer
    }

    if (waferFrameCount >= REQUIRED_FRAMES) {
      waferDetected = true;
      console.log("WAFER DETECTED ✅ - Now bring it to your mouth!");
      // Set permanent status
      waferStatusEl.textContent = "WAFER DETECTED ✅";
      waferStatusEl.className = "status-value pass";
    } else {
      // Show current confidence and frame count
      waferStatusEl.textContent = `Searching... Wafer: ${(waferProb * 75).toFixed(1)}% (${waferFrameCount}/${REQUIRED_FRAMES})`;
      waferStatusEl.className = "status-value warning";
    }
  } catch (err) {
    console.error("Error in wafer prediction:", err);
  }
}

// ===== MEDIAPIPE SETUP =====
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
faceMesh.onResults((results) => { latestFace = results; });

// Hands setup (SINGLE HAND)
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions({
  maxNumHands: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
  modelComplexity: 1,
});
hands.onResults((results) => { latestHands = results; });

// ===== CAMERA LOOP =====
const cam = new Camera(video, {
  onFrame: async () => {
    // Ensure canvas matches current video frame
    if (video.videoWidth && video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Send frame to both MediaPipe models (only if wafer detected)
    if (waferDetected) {
      await faceMesh.send({ image: video });
      await hands.send({ image: video });
    }

    // Run wafer detection only until wafer is detected
    if (!waferDetected) {
      await predictWafer();
    }

    // Draw base video frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Face/Hand presence detection (only if wafer detected)
    const faceOk = waferDetected ? !!latestFace?.multiFaceLandmarks?.length : false;
    const handOk = waferDetected ? !!latestHands?.multiHandLandmarks?.length : false;
    
    // Update status elements
    faceStatusEl.textContent = faceOk ? "✅" : "❌";
    faceStatusEl.className = faceOk ? "status-value pass" : "status-value fail";
    
    handStatusEl.textContent = handOk ? "✅" : "❌";
    handStatusEl.className = handOk ? "status-value pass" : "status-value fail";

    // Draw status on canvas
    ctx.fillStyle = "#ffffff";
    ctx.font = "14px system-ui";
    
    if (!waferDetected) {
      ctx.fillText("Step 1: Show wafer to camera", 10, 20);
    } else {
      ctx.fillText(`Face: ${faceOk ? "✅" : "❌"}  Hand: ${handOk ? "✅" : "❌"}`, 10, 20);
      if (!waferTaken) {
        ctx.fillStyle = "#ffd24d";
        ctx.font = "bold 16px system-ui";
        ctx.fillText("Step 2: Bring wafer to your mouth", 10, 40);
      } else if (!eatingDetected) {
        ctx.fillStyle = "#ffd24d";
        ctx.font = "bold 16px system-ui";
        ctx.fillText("Step 3: Start eating the wafer", 10, 40);
      }
    }

    // LIPS: compute lip center & openness (only if wafer detected)
    let lipCenter = null;
    let openness = 0;
    if (faceOk && waferDetected) {
      const lm = latestFace.multiFaceLandmarks[0];
      const lips = getLipPoints(lm, canvas.width, canvas.height);
      lipCenter = getCenterPoint(lips);

      // Chew detection (open/close cycles)
      const pUpper = { x: lm[13].x * canvas.width,  y: lm[13].y * canvas.height };
      const pLower = { x: lm[14].x * canvas.width,  y: lm[14].y * canvas.height };
      const pLeft  = { x: lm[61].x * canvas.width,  y: lm[61].y * canvas.height };
      const pRight = { x: lm[291].x * canvas.width, y: lm[291].y * canvas.height };

      const mouthWidth = Math.max(1, euclideanDistance(pLeft, pRight));
      const mouthGap   = euclideanDistance(pUpper, pLower);
      openness         = mouthGap / mouthWidth;

      const ts = performance.now();
      if (mouthState === "closed" && openness > OPEN_THR) {
        mouthState = "open";
      } else if (mouthState === "open" && openness < CLOSE_THR) {
        mouthState = "closed";
        if (waferTaken) pushChewEvent(ts);
      }
    }

    // HAND (single): gating via band check (only if wafer detected)
    let indexDistance = null, thumbDistance = null;
    if (handOk && lipCenter && waferDetected) {
      const handLm = latestHands.multiHandLandmarks[0];
      const indexTip = { x: handLm[8].x * canvas.width, y: handLm[8].y * canvas.height };
      const thumbTip = { x: handLm[4].x * canvas.width, y: handLm[4].y * canvas.height };

      indexDistance = euclideanDistance(lipCenter, indexTip);
      thumbDistance = euclideanDistance(lipCenter, thumbTip);

      const indexInRange = (indexDistance >= TOUCH_MIN_PX && indexDistance <= TOUCH_MAX_PX);
      const thumbInRange = (thumbDistance >= TOUCH_MIN_PX && thumbDistance <= TOUCH_MAX_PX);
      const holding = indexInRange && thumbInRange;

      const now = performance.now();
      if (!waferTaken) {
        if (holding) {
          if (!holdingPrev) console.log("Hold started (tips near lips)");
          holdingPrev = true;
          if (contactStartTs == null) contactStartTs = now;
          lastHoldMs = now - contactStartTs;
          if (lastHoldMs >= CONTACT_REQUIRED_MS) {
            waferTaken = true;
            console.log("WAFER TAKEN TO MOUTH ✅ (chew counting active)");
          }
        } else {
          if (holdingPrev) console.log("Hold reset");
          holdingPrev = false;
          contactStartTs = null;
          lastHoldMs = 0;
        }
      }

      // Labels for distances and hold progress (only show if wafer detected)
      ctx.fillStyle = "#ffffff";
      ctx.font = "14px system-ui";
      ctx.fillText(`Index distance: ${indexDistance.toFixed(1)} px`, 10, canvas.height - 60);
      ctx.fillText(`Thumb distance: ${thumbDistance.toFixed(1)} px`, 10, canvas.height - 40);

      if (!waferTaken && holding) {
        const secs = Math.min(CONTACT_REQUIRED_MS, lastHoldMs) / 1000;
        ctx.fillStyle = "#ffd24d";
        ctx.font = "bold 16px system-ui";
        ctx.fillText(`Hold near lips: ${secs.toFixed(1)} / ${(CONTACT_REQUIRED_MS/1000).toFixed(1)} s`, 10, 60);
      }
    } else {
      // If no hand or no lips, reset contact timer (but keep waferTaken once true)
      if (contactStartTs !== null) console.log("Hold reset (lost hand/face)");
      contactStartTs = null;
      lastHoldMs = 0;
      holdingPrev = false;
    }

    // Status labels on canvas (only if wafer detected)
    if (waferDetected) {
      if (waferTaken) {
        ctx.fillStyle = "#7cff8e";
        ctx.font = "bold 16px system-ui";
        ctx.fillText("WAFER TAKEN TO MOUTH ✅ (chew counting active)", 10, 80);
      }
      if (lipCenter) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "14px system-ui";
        ctx.fillText(`Chews (last ${EAT_WINDOW_MS/1000}s): ${chewEvents.length}`, 10, canvas.height - 20);
        ctx.fillText(`Mouth openness: ${openness.toFixed(3)}`, 10, canvas.height - 80);
        if (eatingDetected) {
          ctx.fillStyle = "#00ffa6";
          ctx.font = "bold 18px system-ui";
          ctx.fillText("EATING ✅ - Process Complete!", 10, canvas.height - 100);
        }
      }
    }

    // Update status panel
    updateStatusElements();
    
    // Ensure wafer status remains permanent once detected
    if (waferDetected) {
      waferStatusEl.textContent = "WAFER DETECTED ✅";
      waferStatusEl.className = "status-value pass";
    }
  },
  width: 640,
  height: 480,
});

// ===== INITIALIZATION =====
async function init() {
  try {
    // Initialize wafer detection model
    await initWaferModel();
    
    // Start camera
    await cam.start();
    console.log("Combined detection system initialized successfully");
  } catch (err) {
    console.error("Error initializing system:", err);
  }
}

// Start the system
init();