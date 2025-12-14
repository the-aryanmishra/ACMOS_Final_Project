import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting

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
# Note: Power is now an OBJECTIVE, not just a constraint. 
# However, we can still bound the problem or filter results later.

# VOLTAGE SWEEP (1.8V -> 0.4V)
VOLTAGE_SWEEP = [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================
def run_ltspice(w_in, w_load, w_tail, b_ratio, i_bias, vdd):
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

    # --- D. Parse Log ---
    gain_db = 0.0
    bw_hz = 0.0
    power_current = 1.0 

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='latin-1', errors='ignore') as f:
                log = f.read()
            
            # Gain
            g_match = re.search(r'\(\s*([0-9.-]+)dB', log, re.IGNORECASE)
            if g_match: gain_db = float(g_match.group(1))
            
            # BW Parsing
            b_match = re.search(r'bw:.*AT\s*=?\s*([0-9e\+\-\.]+)', log, re.IGNORECASE)
            if b_match: bw_hz = float(b_match.group(1))
            else:
                b_match_alt = re.search(r'bw\s*=\s*([0-9e\+\-\.]+)', log, re.IGNORECASE)
                if b_match_alt: bw_hz = float(b_match_alt.group(1))
            
            # Itotal
            i_match = re.search(r'itotal\s*[:=]\s*AVG.*=\s*([0-9e\+\-\.]+)', log, re.IGNORECASE)
            if i_match: power_current = abs(float(i_match.group(1)))
                
        except Exception:
            pass

    # --- E. CLEANUP ---
    for filename in [NETLIST_FILE, LOG_FILE, "run.raw", "run.op"]:
        if os.path.exists(filename):
            try: os.remove(filename)
            except: pass

    power_watts = power_current * vdd
    return gain_db, bw_hz, power_watts

# ==========================================
# 3. OPTIMIZATION PROBLEM (Multi-Objective)
# ==========================================
class OTAOptimization(Problem):
    def __init__(self, current_vdd):
        self.current_vdd = current_vdd
        # n_obj=2: 1. Maximize Gain (-Gain), 2. Minimize Power
        # n_constr=2: 1. Min Gain, 2. Min BW
        super().__init__(n_var=5, n_obj=2, n_constr=2, 
                         xl=np.array([0.5,  0.5,  0.5,  0.5, 0.01]), 
                         xu=np.array([10.0, 5.0,  10.0, 5.0, 5.0]))

    def _evaluate(self, X, out, *args, **kwargs):
        F = [] # Objectives
        G = [] # Constraints
        
        for row in X:
            w_in, w_load, w_tail, b, i_bias = row
            gain, bw, power = run_ltspice(w_in, w_load, w_tail, b, i_bias, self.current_vdd)
            
            # --- OBJECTIVES ---
            # 1. Minimize negative gain (Maximize Gain)
            # 2. Minimize Power
            f1 = -gain 
            f2 = power 
            
            # --- CONSTRAINTS (G <= 0 is feasible) ---
            # 1. Gain must be > LIMIT_GAIN_MIN
            #    (LIMIT - gain) <= 0  ->  gain >= LIMIT
            g1 = LIMIT_GAIN_MIN - gain
            
            # 2. BW must be > LIMIT_BW_MIN
            g2 = LIMIT_BW_MIN - bw
            
            # Note: We removed the Power constraint to allow the Pareto front 
            # to explore the trade-off, but the genetic algorithm will naturally
            # push for lower power because it's an objective (f2).

            F.append([f1, f2])
            G.append([g1, g2])

        out["F"] = np.array(F)
        out["G"] = np.array(G)

# ==========================================
# 4. MAIN RUN
# ==========================================
if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"STARTING PARETO ANALYSIS (Gain vs Power)")
    print(f"{'='*60}\n")

    for vdd in VOLTAGE_SWEEP:
        print(f"--- Generating Pareto for VDD = {vdd} V ---")
        
        problem = OTAOptimization(current_vdd=vdd)
        
        # NSGA2 is designed exactly for this (Pareto Fronts)
        algorithm = NSGA2(pop_size=40,    # Increased slightly for better fronts
                          n_offsprings=20,
                          sampling=FloatRandomSampling(),
                          crossover=SBX(prob=0.9, eta=15),
                          mutation=PM(eta=20),
                          eliminate_duplicates=True)

        res = minimize(problem, algorithm, ('n_gen', 30), verbose=False)

        # --- EXTRACT PARETO FRONT ---
        # res.F returns the objective values of the optimal set
        # F[:, 0] is -Gain, F[:, 1] is Power
        if res.F is not None and len(res.F) > 0:
            
            # Convert back to real units
            pareto_gain = -res.F[:, 0]  # Flip sign back to positive dB
            pareto_power_uW = res.F[:, 1] * 1e6 # Convert to uW
            
            # Sort by Power for a cleaner line plot
            sorted_indices = np.argsort(pareto_power_uW)
            pareto_power_uW = pareto_power_uW[sorted_indices]
            pareto_gain = pareto_gain[sorted_indices]

            # --- PLOTTING ---
            plt.figure(figsize=(10, 6))
            plt.plot(pareto_power_uW, pareto_gain, '-o', color='b', mfc='r')
            
            plt.title(f"Pareto Frontier for Vdd = {vdd}V")
            plt.xlabel("Power (uW) [Minimize]")
            plt.ylabel("Gain (dB) [Maximize]")
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # Save Image
            filename = f"pareto_for_{vdd}V.png"
            plt.savefig(filename)
            plt.close() # Close figure to free memory
            
            print(f"  [SAVED] {filename} with {len(pareto_gain)} points.")
            
            # Print the best "compromise" (just the one with max gain for sanity check)
            max_gain_idx = np.argmax(pareto_gain)
            print(f"  [INFO] Max Gain found: {pareto_gain[max_gain_idx]:.2f}dB @ {pareto_power_uW[max_gain_idx]:.2f}uW")
            
        else:
            print(f"  [FAIL] No feasible solutions found for VDD={vdd}")

    print("\nDone. Check the folder for .png files.")