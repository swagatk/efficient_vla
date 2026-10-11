import inspect
import lerobot.scripts.lerobot_eval as le

print("Source of lerobot_eval.eval_policy:")
if hasattr(le, "eval_policy"):
    print(inspect.getsource(le.eval_policy))
elif hasattr(le, "eval"):
    print(inspect.getsource(le.eval))
