# To-DO list

1. use temporal window for gate model in phase 2. --> completed 21/7/2026
2. Train phase 3 with L2 penalty for the delta action with anchored-demonstration. --> completed 22/7/2026
3. Run phase 4 sweep in delta mode for tasks = [1, 2, 4, 6, 7, 8] for 3 seeds to find best values for threshold and alpha. --> Currently running
4. Run phase 4 sweep in absolute mode for tasks = [1, 2, 4, 6, 7, 8] for 3 seeds to find best values for threshold and alpha. --> completed 23/7/2026
5.  Once the best threshold and alpha are chosen from the sweep, run TASKS="0 1 2 3 4 5 6 7 8 9" once to generate the final, clean aggregate report across all 10 tasks for publication/documentation using 'run_phase4_eval.sh' for both delta and absolute mode.