import subprocess
import os
import atexit # Not strictly needed if using

class LinuxInhibit:
    def __init__(self, reason="RL Training"):
        self.reason = reason
        self.process = None
        self.orig_profile = None

    def __enter__(self):
        # We use gnome-session-inhibit to create a lock
        # Flags: 4 = Inhibit suspend, 8 = Inhibit idle
        command = [
            "gnome-session-inhibit",
            "--reason", self.reason,
            "--inhibit", "suspend",
            "--app-id", "PythonRLScript",
            "sleep", "infinity"
        ]
        try:
            self.process = subprocess.Popen(command)
            print(f"--- Power Lock Active: {self.reason} ---")
        except FileNotFoundError:
            print("gnome-session-inhibit not found. Power lock skipped.")
            
        try:
            res = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True)
            self.orig_profile = res.stdout.strip()
            subprocess.run(["powerprofilesctl", "set", "performance"], check=True)
            print(f"--- Power profile set to 'performance' (was '{self.orig_profile}') ---")
        except Exception as e:
            print(f"Could not set power profile: {e}")
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            print("--- Power Lock Released ---")
            
        if self.orig_profile:
            try:
                subprocess.run(["powerprofilesctl", "set", self.orig_profile], check=False)
                print(f"--- Power profile restored to '{self.orig_profile}' ---")
            except Exception as e:
                print(f"Could not restore power profile: {e}")

# --- HOW TO USE IT ---
if __name__ == "__main__":
    with LinuxInhibit("Training Alienware Model"):
        print("Starting heavy GPU task...")
        # Put your training loop or VS Code execution here
        # ...