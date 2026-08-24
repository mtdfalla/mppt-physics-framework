"""
study_temperature.py — Temperature-sensitivity study (round-2 revision)
=======================================================================
Re-runs the full 8-profile x 3-algorithm matrix at elevated cell
temperatures for the PSC level given on the command line. The baseline
matrix at 25 C is the main 96-case set. Elevated temperatures enter the
single-diode model through the thermal voltage, the saturation current,
the photocurrent temperature coefficient and the Voc coefficient, all
already implemented in tct_eval.py.

Usage:
    python3 study_temperature.py --T 45 --psc easy
    python3 study_temperature.py --T 65 --psc extreme

Output: results/study_temperature_T{T}_{psc}.json  (run_psc format)
"""
import argparse, os, sys

sys.stdout.reconfigure(encoding='utf-8')

import modular_test_runner as mtr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, required=True, help='cell temperature C')
    ap.add_argument('--psc', required=True, choices=list(mtr.PSC_PATTERNS))
    a = ap.parse_args()
    mtr.T_CELSIUS = a.T          # run_psc reads this module-level constant
    out_file = os.path.join(mtr.RESULTS_DIR,
                            f'study_temperature_T{int(a.T)}_{a.psc}.json')
    mtr.run_psc(a.psc, out_file)

if __name__ == '__main__':
    main()
