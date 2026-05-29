* If you want, next I can add a training-time auxiliary anchor loss that keeps diffusion predictions close to base-policy actions early in training and decays over epochs; that usually stabilizes early hybrid learning.

Suggested next plan (priority order)

1. Do a tight alpha sweep first
    * Try residual alpha: 0.02, 0.05, 0.1, 0.2, 0.3
    * Keep your current safety settings on
    * Use at least 30 episodes per task for stability
    * Goal: check whether 0.1 is already too aggressive

2. Make it task-adaptive immediately
    * If task 3 benefits but task 0/1 collapse, run:
        * Base-only on weak tasks
        * Residual only on tasks where it helps
    * This gives practical gains now while you keep researching
3. Retrain as true residual target (most important model change)
    * In training, predict delta action instead of absolute action:
        * delta target = dataset action - base action on same observation
        * inference action = base + alpha × predicted delta
    * Add a penalty to keep delta small (L2 on delta magnitude)
    * This aligns training with what worked in eval
    * Main files: train_hybrid_diffusion.py, hybrid_diffusion_agent.py
4. Fix train-eval mismatch around control horizon
    * Reduce chunk size for eval and training (for example 1 to 4 instead of long open-loop chunks)
    * Re-check control frequency assumption used by delta timestamps in train_hybrid_diffusion.py:862
    * Long open-loop chunks are usually fragile for LIBERO contact tasks
5. Select best checkpoint by eval, not last epoch
    * You already saw non-monotonic behavior before
    * Save and compare top-k checkpoints by eval success
    * Do not assume epoch 20 is best
6. Add one hard stop criterion
    * If after residual-target retraining plus alpha sweep the best score is still below 0.45 to 0.50, pause this branch and shift effort to stronger baseline-preserving methods (for example RL residual or task-specific adapters)
