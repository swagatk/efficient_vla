# Novel Research Directions for Edge VLA Training

Given the 16GB VRAM constraint and the limited improvements seen with the Gated Residual Strategy, exploring non-parametric, inference-time, or highly structured adaptation methods is the best path forward. These approaches directly address the failures of monolithic parametric correctors (e.g., overfitting, catastrophic forgetting) while fitting comfortably on a consumer laptop.

## 1. Representation Engineering (RepE) / Activation Steering for VLAs
Instead of training a neural network to predict a "delta action" at the output layer (which failed in the Phase 4 eval), directly steer the VLM's "thoughts" in its latent space.

*   **The Idea:** In LLMs, "Activation Addition" is used to steer models (e.g., making them less toxic) by finding a vector direction in the hidden states that represents a concept, and adding it during inference. You can apply this to robotics. Use the Phase 1 dataset to extract the hidden activations of SmolVLA on *successful* trajectories and *failed* trajectories. Compute the mean difference vector between them: $\vec{v}_{steer} = \mu_{success} - \mu_{failure}$. During deployment, simply add $\alpha \times \vec{v}_{steer}$ to the VLM's hidden states at inference time.
*   **Why it's publishable:** RepE is a massive trend in LLMs but is almost entirely unexplored in end-to-end continuous control / VLAs.
*   **16GB Constraint:** Perfect. It requires **zero** additional parameters to be trained via backprop. You just run forward passes to collect activations, run PCA or mean pooling on CPU, and do vector addition at test time.

## 2. Test-Time Compute (TTC) / Inference-Time Action Search
Recent breakthroughs (like OpenAI's o1) rely heavily on test-time compute—spending more time "thinking" before acting. Most VLAs just take the single argmax action, which is fragile.

*   **The Idea:** Train a tiny "Action Evaluator" or Energy-Based Model (EBM) that takes in the current image and a proposed action sequence, and outputs a success probability. At test time, use SmolVLA to sample $N$ different action trajectories (using temperature scaling or dropout). Score all $N$ trajectories using your Evaluator, and execute the one with the highest score.
*   **Why it's publishable:** Test-time optimization in vision-language-action models is highly novel. You can present a compelling Pareto curve showing how trading off latency (sampling more actions) strictly increases the success rate.
*   **16GB Constraint:** Excellent. You freeze SmolVLA entirely. The Evaluator can be a very lightweight frozen CNN (like ResNet-18) or a tiny MLP trained on your Phase 1 success/failure dataset using Contrastive Learning. It will use $< 2$GB of VRAM.

## 3. Retrieval-Augmented Control (RAC) via Non-Parametric Memory
Your Gated Residual Corrector tried to memorize corrections in its weights, which likely caused overfitting.

*   **The Idea:** Maintain a "Memory Bank" of the visual embeddings and successful actions from your LIBERO human demonstrations. During evaluation, use a lightweight embedding network to compute the cosine similarity between the robot's current visual state and the memory bank. If the VLA predicts an action that heavily diverges from the retrieved nearest-neighbor's action, you dynamically interpolate the VLA's action with the retrieved action.
*   **Why it's publishable:** RAG for robotics. It solves the catastrophic forgetting problem of RL without needing backpropagation. You can claim it as a "Self-Correcting Edge VLA via Non-Parametric Memory."
*   **16GB Constraint:** Highly efficient. The memory bank is stored in CPU RAM. The nearest neighbor search is cheap. You only need to train a tiny contrastive encoder (like a 10M parameter CNN) to map images to the retrieval latent space.

## 4. Dynamic Phase-Aware LoRA Routing (MoE)
Your Gated Residual Strategy might have failed because the corrector had to fix errors across entirely different phases of a task (reaching, grasping, manipulating).

*   **The Idea:** Train a tiny classifier to predict the current "sub-task phase" (e.g., Phase 1: Approaching, Phase 2: Grasping, Phase 3: Placing). Then, instead of one monolithic corrector, train 3 separate, extremely low-rank LoRA adapters (e.g., rank $r=2$) inside the SmolVLA backbone. At inference time, dynamically route the forward pass through the LoRA adapter specific to the current phase.
*   **Why it's publishable:** Introduces temporal/hierarchical structure to VLA fine-tuning. It shows that VLAs suffer from "capacity interference" between reaching and manipulating, and dynamic LoRA routing solves it.
*   **16GB Constraint:** Very feasible. QLoRA/LoRA fine-tuning of a 500M parameter model with rank 2 easily fits in 8-10GB of VRAM.
