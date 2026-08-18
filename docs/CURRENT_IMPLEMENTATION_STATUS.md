# Smart Workplace Safety — Current Implementation Status

## 1. Real AI models in the runtime

The project does **not** use manual boxes as AI proof.

### Primary PPE model
`models/candidates/prodbykosta_ppe_best.pt`

Purpose: detect `Person` and positive PPE evidence such as Helmet, Vest, Glove, Boots, Mask and Glass.

### Construction model
`models/candidates/snehil_yolov8n_ppe_best.pt`

Purpose: detect Person, positive/negative PPE, `machinery`, `vehicle` and Safety Cone.

### Auxiliary hazard model
`best.pt`

Purpose: supplementary AI evidence for classes including `fall_hazard`, `unsafe_ladder_use`, `phone-usage`, `no-fall-protection`, and `construction-machine`.

Important: the auxiliary model is **not described as a temporal Action Recognition model**. Because an earlier real project sample produced a one-frame `fall_hazard` false positive, unsafe-behavior evidence from this model now requires **3 positive inference observations in the last 5 inference frames** before becoming a confirmed behavior hazard.

## 2. False-positive controls

- Helmet is marked SAFE only when both PPE models agree spatially and the agreement persists for at least 3 of 5 inference frames.
- Vest uses the same two-model + temporal rule.
- One missing positive detection never means missing PPE by itself.
- Explicit negative classes from the construction model may generate PPE violation evidence.
- A one-frame auxiliary hazard (`fall_hazard`, etc.) does not create a confirmed behavior incident.
- Incident cooldown prevents the database from recording the same tracked hazard on every frame.

## 3. Worker / equipment relationships

The AI models provide actual object boxes and confidences. Tracking associates person boxes over adjacent frames. The system then computes:

- Worker track ID
- Person model confidence
- Equipment class and model confidence
- Worker foot point
- Dynamic equipment safety buffer
- `CLOSE_TO` relationship when the tracked worker enters that buffer
- Worker↔equipment line
- Pixel distance between worker foot point and equipment center

**Distance is reported in pixels only.** The system does not claim meters until camera calibration + ground-plane homography are configured.

## 4. Safety zones

### Fixed restricted zones
The supervisor can define a normalized polygon per camera. Zones are persisted in SQLite. The tracked worker foot point is tested against the polygon. Entry creates `RESTRICTED_ZONE_ENTRY` (HIGH).

### Dynamic danger zones
A dynamic safety buffer is generated around detected machinery/vehicle. Persistent worker proximity (3/5 inference frames) creates `DANGEROUS_MACHINE_PROXIMITY` (CRITICAL).

## 5. Incidents and reporting

SQLite stores:
- timestamp
- source/video/live camera
- frame index
- worker track ID
- hazard type
- severity
- confidence
- JSON details

APIs expose recent incidents and aggregate reports by severity and hazard type.

## 6. Supervisor functions

- Session login: implemented
- Dashboard: implemented
- Image analysis: implemented
- Uploaded video analysis: implemented
- Live webcam/RTSP endpoint: implemented (`CAMERA_SOURCE`)
- Restricted-zone create/list/delete: implemented
- Incident review: implemented
- Report summary: implemented
- Emergency action: application-level simulation implemented

The emergency endpoint must **not** be presented as physical PLC/machine shutdown unless hardware integration is added.

## 7. Functional requirement matrix

| Requirement | Status | Evidence / implementation |
|---|---|---|
| Secure supervisor login | Implemented | Flask session + protected routes |
| Workplace monitoring UI | Implemented | Arabic RTL dashboard |
| Immediate alerts | Implemented | HIGH/CRITICAL worker state + incident generation |
| Review unsafe situations | Implemented | SQLite incident list |
| Safety reports / summaries | Implemented | report summary API + dashboard |
| Emergency response | Implemented as application simulation | `/api/emergency` |
| Capture/process uploaded video | Implemented | `/analyze-video` |
| Live webcam/RTSP | Implemented, requires configured camera source for E2E proof | `/live-feed`, `CAMERA_SOURCE` |
| Continuous monitoring | Implemented in live stream loop | sampled real inference |
| Detect workers | Real AI | two selected PPE/construction models |
| Track workers | Implemented | IoU temporal tracker |
| Define virtual safety zones | Implemented | SQLite polygon zones |
| Restricted-area entry | Implemented | tracked foot-point in polygon |
| Dangerous worker-machine proximity | Implemented | dynamic zone + temporal confirmation |
| PPE violations | Implemented with multi-model safeguards | explicit negative + agreement rules |
| Unsafe hazard/behavior evidence | Partially implemented | real auxiliary hazard detector + 3/5 temporal filter |
| Dedicated sequence Action Recognition model | **Not yet integrated** | official construction model weight found externally, not stored in GitHub |
| Store incidents | Implemented | SQLite |
| Reports | Implemented | aggregation APIs/UI |

## 8. Current proof limitation

The three original project videos exist in the ChatGPT execution environment, while the real model binaries are currently executed through the GitHub project/runner. Full 3-video real-model processing should only be marked PASS after the original video bytes are available to the environment that holds the real weights. Existing manually annotated prototype videos must not be used as final AI-inference evidence.
