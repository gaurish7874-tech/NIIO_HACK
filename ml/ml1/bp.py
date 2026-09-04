"""
bp.py
-----
Estimates Blood Pressure (Systolic and Diastolic) from Heart Rate.
This is a heuristic/mock approach suitable for hackathon demonstrations,
as purely contactless BP from webcams without calibration remains an unsolved research challenge.
"""

def estimate_blood_pressure(heart_rate: float) -> tuple[int, int]:
    """
    Estimates Blood Pressure based on current Heart Rate.
    Normal resting HR is ~60-80 BPM, Normal BP is ~120/80.
    As HR increases, BP typically increases as well.
    
    Returns:
        tuple: (systolic, diastolic)
    """
    if not heart_rate or heart_rate <= 0:
        return (120, 80)
        
    # Heuristic mapping:
    # Baseline: HR 60 -> BP 110/70
    # Every 10 BPM increase roughly adds ~6 mmHg systolic and ~3 mmHg diastolic
    sys = 110 + ((heart_rate - 60) * 0.6)
    dia = 70 + ((heart_rate - 60) * 0.3)
    
    # Cap values to realistic ranges
    sys = max(90, min(180, sys))
    dia = max(60, min(110, dia))
    
    return int(round(sys)), int(round(dia))
