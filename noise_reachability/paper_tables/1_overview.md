| Exp. | Question | Episodes | Outcome |
| --- | --- | --- | --- |
| E1 | Per-state success over 16 seeds | 700 | 8 of 50 states at 0 |
| E2 | Training-free chunk selection | 900 | negative |
| E3 | Learned noise critics | 4,120 | not read |
| E4 | LoRA on the policy's own successes | 1,430 | negative |
| E5a | Fixed noise on task 0's systematic states | 1,249 | all 5 reached |
| E5b | Learned per-episode noise | 5,280 | did not learn |
| E6 | Do the labels hold on fresh seeds? | 1,280 | yes, 1 of 48 |
| E7 | Fixed noise beyond task 0 | 1,095 | failed; labels too coarse |
| E8 | Fixed noise on resistant states | 3,172 | both tests pass |
| E9 | Do found noises transfer between states? | 662 | not shown |
| E10 | Distil noise successes into weights | 2,940 | both tests pass |
| E11 | Replicate E8 on eight new states | 520 | both tests pass |
| E12 | Second round of search and training | -- | running |
