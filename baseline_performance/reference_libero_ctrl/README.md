# External reference: SmolVLA on LIBERO-10 from LIBERO-CTRL

`smolvla_libero_10_clean.csv` holds the 500 unperturbed LIBERO-10 rollouts (10 tasks x all 50 init states,
one rollout each) of SmolVLA from:

- H. Sawada and S. Kasahara, "Beyond Single-Axis Testing: Paired Evaluation of Compound Robustness in
  Vision-Language-Action Policies", arXiv:2609.15940 (2026).
- Code and per-rollout records: https://github.com/hiroki-oist/LIBERO-ctrl (MIT licence), commit
  `2c866ac70b4efda3bfb641bdfd9c67c97b383628`, file `results/paper/smolvla_clean/rollouts.jsonl`
  (sha256 `5be2d239e8b0d2a99586c8bd2a3ce64aafcf44a1cbd6a9e77975016b8eaa8974`), rows with
  `suite == "libero_10"`; `init_id` is written here as `init_state`. (The `policy` field in that file says
  `oft` for every row; the directory and the paper identify the policy as SmolVLA.)

Their setup: checkpoint `lerobot/smolvla_libero` (weights sha256
`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`, identical in Hub revisions `31d453f7ed`
and `a24f1f3804`), `n_action_steps=1`, LeRobot's own pre/post-processors and LIBERO processor step
(180-degree image rotation, 8-dimensional state), lerobot 0.6.1, 256x256 cameras, horizon 520 steps.
They report 76.35% over all four suites against the paper's 87.3%, and state that the released checkpoint
was trained on one eighth of the published sample budget (25,000 steps at batch size 32).

This file: 284/500 (56.8%); on init states 0-9 only, 59/100.

**This is not the checkpoint behind `results_300926`.** That run used `HuggingFaceVLA/smolvla_libero`
(September 2025), a different model: action expert trained with a frozen VLM and vision encoder
(`train_expert_only=true`, `expert_width_multiplier=0.5`, all VLM layers, SmolVLM2-500M-Instruct), weights
sha256 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`. `lerobot/smolvla_libero`
(March 2026) fine-tunes the whole model (`train_expert_only=false`, 16 VLM layers, expert width 0.75,
SmolVLM2-500M-Video-Instruct) and names its cameras `camera1`/`camera2`.
