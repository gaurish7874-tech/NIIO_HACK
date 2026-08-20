"""
demo_live.py
------------
End-to-End Live Demonstration of the Contactless Multimodal Wellness Monitor.

This script connects ML-1 (Webcam -> Vitals) directly to ML-2 (Vitals -> Triage/RAG).
It captures 300 frames (10 seconds) of webcam video, extracts physiological and
behavioral signals, and then runs the AI Agentic Debate to give you a wellness score
and RAG-grounded guidance based on your actual live readings.
"""

import os
import sys
import cv2
import time
import logging
from dotenv import load_dotenv

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.ml1.multimodal import analyze_multimodal_wellness
from ml.ml2.triage import predict_triage_score

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

# Reduce logging noise for the demo
logging.getLogger("ml2").setLevel(logging.WARNING)

def main():
    print("===============================================================")
    print("  NIIO HACK — End-to-End Live Triage Demo")
    print("===============================================================")
    print("Initializing webcam... Please sit still in a well-lit area.")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}.")
        return

    # Cap framerate to 30 FPS for consistent rPPG processing
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    
    frames = []
    buffer_size = 300  # 10 seconds of video needed for rPPG heart rate

    print("\n[PHASE 1: ML-1 Signal Extraction]")
    print("Capturing 10 seconds of video to calculate heart rate and blinks...")
    
    start_time = time.time()
    
    while len(frames) < buffer_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        frames.append(frame)
        
        # UI Feedback
        progress = len(frames) / buffer_size * 100
        
        # Draw progress bar on frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Analyzing: {int(progress)}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("NIIO HACK - Vitals Extraction", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
            
    end_time = time.time()
    actual_fps = len(frames) / (end_time - start_time)
    print(f"Captured {len(frames)} frames at ~{actual_fps:.1f} FPS.")
    
    # Cleanup camera
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nExtracting Multimodal Vitals (ML-1) (This takes a few seconds)...")
    vitals = analyze_multimodal_wellness(frames, fps=actual_fps)
    
    print("\n--- Live Vitals Captured ---")
    print(f"Heart Rate: {vitals['physio'].get('hr', 'N/A')} BPM")
    print(f"HRV:        {vitals['physio'].get('hrv', 'N/A')} ms")
    print(f"Respiration:{vitals['physio'].get('respiration', 'N/A')} RPM")
    print(f"Blink Rate: {vitals['behavioral'].get('blink_rate', 'N/A')} /min")
    
    print("\n[PHASE 2: ML-2 Enriched Triage Engine]")
    print("Running AI Agentic Debate (Physio Agent vs Behavioral Agent)...")
    
    # Run ML-2 Pipeline
    result = predict_triage_score(vitals)
    
    print("\n===============================================================")
    print("  LIVE TRIAGE RESULTS")
    print("===============================================================")
    print(f"Verdict: {result.verdict}")
    print(f"Summary: {result.summary}")
    
    if result.wellness_score:
        ws = result.wellness_score
        print(f"\nWellness Score: {ws['score']}/100 ({ws['category']})")
        
    if result.contradiction and result.contradiction.get("has_contradiction"):
        ct = result.contradiction
        print(f"\nContradiction Detected: {ct['label']}")
        print(f"Analysis: {ct['explanation']}")
        
    if result.emotional_map:
        em = result.emotional_map
        print(f"\nEmotional State: {em['quadrant_label']}")
        print(f"Arousal: {em['arousal']}, Valence: {em['valence']}")
        
    if result.guidance:
        # Safely encode for windows console
        safe_guidance = result.guidance.encode("ascii", errors="replace").decode("ascii")
        print(f"\n--- RAG AI Guidance ---\n{safe_guidance}")
        print(f"Sources: {result.guidance_sources}")

if __name__ == "__main__":
    main()
