import os
import subprocess
import re
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# ==========================================
# 1. CONFIGURATION
# ==========================================
LTSPICE_CMD = r"C:\Program Files\ADI\LTspice\LTspice.exe"
TEMPLATE_FILE = "ota_template.sp"
NETLIST_FILE  = "run.net"
LOG_FILE      = "run.log"

# CONSTRAINTS
LIMIT_GAIN_MIN = 10.0      # dB
LIMIT_BW_MIN   = 10000.0   # 10 kHz
LIMIT_POWER_MAX = 10e-6    # 10 uW

# VOLTAGE SWEEP (1.8V -> 0.4V)
VOLTAGE_SWEEP = [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================
def run_ltspice(w_in, w_load, w_tail, b_ratio, i_bias, vdd):
    """
    Injects genes AND VDD into the SPICE template.
    Uses YOUR EXACT parsing logic.
    """
    # --- A. Read Template ---
    try:
        with open(TEMPLATE_FILE, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return 0, 0, 1.0

    # --- B. Inject Genes & Voltage ---
    content = content.replace("{WIN}",     f"{w_in:.2f}u")
    content = content.replace("{WLOAD}",   f"{w_load:.2f}u")
    content = content.replace("{WTAIL}",   f"{w_tail:.2f}u")
    content = content.replace("{B_RATIO}", f"{b_ratio:.2f}")
    content = content.replace("{IBIAS}",   f"{i_bias:.2f}u")
    
    # NEW: Inject Voltage
    content = content.replace("{VDD_VAL}", f"{vdd:.2f}")

    with open(NETLIST_FILE, 'w') as f:
        f.write(content)

    # --- C. Run LTSpice ---
    abs_netlist_path = os.path.abspath(NETLIST_FILE)
    cmd = [LTSPICE_CMD, "-b", "-Run", abs_netlist_path]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return 0, 0, 1.0

    # --- D. Parse Log (EXACT COPY OF YOUR LOGIC) ---
    gain_db = 0.0
    bw_hz = 0.0
    power_current = 1.0 

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='latin-1', errors='ignore') as f:
                log = f.read()
            
            # Gain
            g_match = re.search(r'\(\s*([0-9.-]+)dB', log, re.IGNORECASE)
            if g_match:
                gain_db = float(g_match.group(1))
            
            # BW Parsing
            b_match = re.search(r'bw:.*AT\s*=?\s*([0-9e\+\-\.]+)', log, re.IGNORECASE)
            if b_match:
                bw_hz = float(b_match.group(1))
            else:
                b_match_alt = re.search(r'bw\s*=\s*([0-9e\+\-\.]+)', log, re.IGNORECASE)
                if b_match_alt:
                    bw_hz = float(b_match_alt.group(1))
                else:
                    bw_hz = 0.0 # Set to 0 if failed
            
            # Itotal
            i_match = re.search(r'itotal\s*[:=]\s*AVG.*=\s*([0-9e\+\-\.]+)', log, re.IGNORECASE)
            if i_match:
                power_current = abs(float(i_match.group(1)))
                
        except Exception:
            pass

    # --- E. CLEANUP ---
    for filename in [NETLIST_FILE, LOG_FILE, "run.raw", "run.op"]:
        if os.path.exists(filename):
            try: os.remove(filename)
            except: pass

    # Calculate Power (Watts) = Volts * Amps
    power_watts = power_current * vdd
    return gain_db, bw_hz, power_watts

# ==========================================
# 3. OPTIMIZATION PROBLEM
# ==========================================
class OTAOptimization(Problem):
    def __init__(self, current_vdd):
        self.current_vdd = current_vdd
        super().__init__(n_var=5, n_obj=1, n_constr=0, 
                         xl=np.array([0.5,  0.5,  0.5,  0.5, 0.01]), 
                         xu=np.array([10.0, 5.0,  10.0, 5.0, 5.0]))

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            w_in, w_load, w_tail, b, i_bias = row
            gain, bw, power = run_ltspice(w_in, w_load, w_tail, b, i_bias, self.current_vdd)
            
            penalty = 0.0
            
            # 1. Gain Check (> 10dB)
            if gain < LIMIT_GAIN_MIN:
                penalty += 1000 + (LIMIT_GAIN_MIN - gain) * 10
            
            # 2. BW Check (> 10kHz)
            if bw < LIMIT_BW_MIN:
                penalty += 500 + (LIMIT_BW_MIN - bw) * 0.1
                
            # 3. Power Check (< 10uW)
            if power > LIMIT_POWER_MAX:
                penalty += 500 + (power - LIMIT_POWER_MAX) * 1e8

            # FITNESS: Maximize Gain (i.e., Minimize -Gain)
            if penalty > 0:
                f1 = penalty
            else:
                f1 = -gain # Negative gain because pymoo minimizes
            F.append([f1])

        out["F"] = np.array(F)

# ==========================================
# 4. MAIN RUN
# ==========================================
if __name__ == "__main__":
    results_table = []

    print(f"{'='*60}")
    print(f"STARTING VDD SWEEP (1.8V -> 0.4V)")
    print(f"CONSTRAINTS: Gain > {LIMIT_GAIN_MIN}dB | BW > {LIMIT_BW_MIN/1e3}kHz | Power < {LIMIT_POWER_MAX*1e6}uW")
    print(f"{'='*60}\n")

    for vdd in VOLTAGE_SWEEP:
        print(f"--- Optimizing for VDD = {vdd} V ---")
        
        problem = OTAOptimization(current_vdd=vdd)
        algorithm = NSGA2(pop_size=20, n_offsprings=10,
                          sampling=FloatRandomSampling(),
                          crossover=SBX(prob=0.9, eta=15),
                          mutation=PM(eta=20),
                          eliminate_duplicates=True)

        res = minimize(problem, algorithm, ('n_gen', 30), verbose=False)

        # --- FIX FOR SINGLE OBJECTIVE RESULT EXTRACTION ---
        if res.X.ndim == 1:
            # If 1D, pymoo returned the single best solution directly
            best_params = res.X
        else:
            # If 2D, it returned a population, so we find the argmin
            best_idx = np.argmin(res.F.flatten()) 
            best_params = res.X[best_idx]
            
        w_in, w_load, w_tail, b, i_bias = best_params
        
        # Validation Run
        final_gain, final_bw, final_power = run_ltspice(w_in, w_load, w_tail, b, i_bias, vdd)
        
        # Area Calc
        L_IN, L_MIR = 0.5, 1.0
        area = (2*w_in*L_IN) + (2*w_load*L_MIR) + (4*w_load*b*L_MIR) + (w_tail*L_MIR)

        # Feasibility Check
        is_feasible = True
        if final_gain < LIMIT_GAIN_MIN: is_feasible = False
        if final_bw < LIMIT_BW_MIN: is_feasible = False
        if final_power > LIMIT_POWER_MAX: is_feasible = False

        if is_feasible:
            print(f"  [SUCCESS] Gain: {final_gain:.2f}dB | Power: {final_power*1e6:.2f}uW")
            res_entry = {
                "Vdd": vdd, "Result": "Feasible",
                "Gain_dB": final_gain, "Power_uW": final_power * 1e6,
                "BW_kHz": final_bw / 1000.0, "Area_um2": area,
                "Params": f"Win={w_in:.2f}, Wload={w_load:.2f}, B={b:.2f}, Ib={i_bias:.2f}"
            }
        else:
            print(f"  [FAIL] Best solution did not meet constraints (G={final_gain:.1f}dB)")
            res_entry = {
                "Vdd": vdd, "Result": "No Feasible Solution",
                "Gain_dB": "-", "Power_uW": "-", "BW_kHz": "-", "Area_um2": "-", "Params": "-"
            }
        
        results_table.append(res_entry)

    # Final Table
    print("\n\n" + "="*100)
    print(f"{'FINAL SUMMARY REPORT':^100}")
    print("="*100)
    header = f"{'Vdd (V)':<10} | {'Status':<20} | {'Gain (dB)':<10} | {'Power (uW)':<12} | {'BW (kHz)':<10} | {'Area (um2)':<12} | {'Params (Best Solution)'}"
    print(header)
    print("-" * 100)

    for row in results_table:
        if row["Result"] == "Feasible":
            line = (f"{row['Vdd']:<10.1f} | {row['Result']:<20} | {row['Gain_dB']:<10.2f} | "
                    f"{row['Power_uW']:<12.2f} | {row['BW_kHz']:<10.2f} | {row['Area_um2']:<12.2f} | {row['Params']}")
        else:
            line = (f"{row['Vdd']:<10.1f} | {row['Result']:<20} | {'-':<10} | {'-':<12} | {'-':<10} | {'-':<12} | -")
        print(line)
    print("="*100)