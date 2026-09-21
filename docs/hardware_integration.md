# Tactical Radar HUD and Hardware Integration

This document explains the real-time Radar HUD visualization and microcontroller-based hardware trigger system.

## 1. Tactical Radar HUD Display

The radar HUD (`scripts/radar_detect.py`) transforms standard optical camera frames into a tactical split-screen interface displaying:
1. **Left Panel**: Annotated camera feed with color-coded bounding boxes, confidence tags, and FPS counter.
2. **Right Panel**: Circular polar radar display simulating an airspace sector scan with plotted target blips.

```
+------------------------------------+------------------------------------+
|                                    |                 N                  |
|                                    |                 |                  |
|           OPTICAL FEED             |        W --- (  +  ) --- E         |
|      [ Aircrafts / Bird / Drone ]  |                 |                  |
|                                    |                 S                  |
|      Detections: 2 | FPS: 28.4     |   TACTICAL RADAR HUD | TRACKS: 2   |
+------------------------------------+------------------------------------+
```

### Coordinate Transformation

Bounding box centers in camera pixel space `(cx, cy)` are mapped to the polar radar scope using normalized offsets:

$$\Delta x = \frac{c_x - W/2}{W/2}, \quad \Delta y = \frac{c_y - H/2}{H/2}$$

Radar coordinates:
$$X_{\text{radar}} = R_{\text{center}} + \Delta x \cdot R_{\text{scope}}$$
$$Y_{\text{radar}} = R_{\text{center}} + \Delta y \cdot R_{\text{scope}}$$

### Launching the Radar Interface

```bash
# Live webcam (index 0)
python scripts/radar_detect.py --model weights/best.pt --source 0

# Video file demo
python scripts/radar_detect.py --model weights/best.pt --source assets/testvidairtrack.mp4

# Save output video
python scripts/radar_detect.py --model weights/best.pt --source assets/testvidairtrack.mp4 --save assets/radar_demo.mp4
```

---

## 2. Arduino Ultrasonic Sensor Trigger

The hardware listener (`scripts/hardware_trigger.py`) enables autonomous perimeter monitoring. An Arduino connected to an ultrasonic distance sensor (such as HC-SR04) continuously scans for approaching objects. When an object enters proximity, the microcontroller broadcasts a trigger signal over serial.

### Serial Protocol

- **Baud Rate**: 9600 bps
- **Trigger Keyword**: `DETECT`

### Arduino Example Sketch

```cpp
const int trigPin = 9;
const int echoPin = 10;
const int thresholdDistanceCm = 150; // Trigger distance

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH);
  long distanceCm = duration * 0.034 / 2;

  if (distanceCm > 0 && distanceCm < thresholdDistanceCm) {
    Serial.println("DETECT");
    delay(5000); // Cooldown delay
  }
  delay(100);
}
```

### Starting the Listener

```bash
python scripts/hardware_trigger.py --port COM3 --baud 9600 --trigger-cmd scripts/radar_detect.py
```
