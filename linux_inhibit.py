import subprocess
import os
import atexit # Not strictly needed if using
import signal


DEVNULL = subprocess.DEVNULL

class LinuxInhibit:
    def __init__(self, reason="RL Training"):
        self.reason = reason
        self.process = None
        self.orig_profile = None
        self.watchdog = None
        self._previous_handlers = {}

    def _set_profile(self, profile):
        if not profile:
            return
        subprocess.run(["powerprofilesctl", "set", profile], check=False)

    def _release_lock(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=2)
            print("--- Power Lock Released ---")
        self.process = None

    def _acquire_lock(self):
        if self.process and self.process.poll() is None:
            return

        command = [
            "gnome-session-inhibit",
            "--reason", self.reason,
            "--inhibit", "suspend",
            "--app-id", "PythonRLScript",
            "sleep", "infinity"
        ]
        try:
            # Detach stdio so caller-side pipes (e.g., tee) are never held open.
            self.process = subprocess.Popen(
                command,
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
            print(f"--- Power Lock Active: {self.reason} ---")
        except FileNotFoundError:
            self.process = None
            print("gnome-session-inhibit not found. Power lock skipped.")

    def _handle_stop(self, sig, frame):
        self._release_lock()
        self._set_profile(self.orig_profile)
        os.kill(os.getpid(), signal.SIGSTOP)

    def _handle_continue(self, sig, frame):
        self._acquire_lock()
        self._set_profile("performance")
        print("--- Power profile set to 'performance' after resume ---")

    def _install_signal_hooks(self):
        for handled_signal, handler in (
            (signal.SIGTSTP, self._handle_stop),
            (signal.SIGCONT, self._handle_continue),
        ):
            self._previous_handlers[handled_signal] = signal.getsignal(handled_signal)
            signal.signal(handled_signal, handler)

    def _restore_signal_hooks(self):
        for handled_signal, handler in self._previous_handlers.items():
            signal.signal(handled_signal, handler)
        self._previous_handlers.clear()

    def __enter__(self):
        self._acquire_lock()
        self._install_signal_hooks()
            
        try:
            res = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True)
            self.orig_profile = res.stdout.strip()
            self._set_profile("performance")
            print(f"--- Power profile set to 'performance' (was '{self.orig_profile}') ---")
            
            # Spawn a detached bash watchdog to guarantee power profile restoration
            # even if this Python process is hard-killed (SIGKILL) or hung.
            pid = os.getpid()
            inhibit_pid = self.process.pid if self.process else None
            watchdog_script = f"""
            while kill -0 {pid} 2>/dev/null; do
                sleep 1
            done
            if [ -n \"{inhibit_pid or ''}\" ]; then
                kill {inhibit_pid or ''} 2>/dev/null
            fi
            powerprofilesctl set {self.orig_profile} 2>/dev/null
            """
            self.watchdog = subprocess.Popen(
                ["bash", "-c", watchdog_script],
                start_new_session=True,
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
        except Exception as e:
            print(f"Could not set power profile: {e}")
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_signal_hooks()
        self._release_lock()
            
        if self.orig_profile:
            try:
                self._set_profile(self.orig_profile)
                print(f"--- Power profile restored to '{self.orig_profile}' ---")
            except Exception as e:
                print(f"Could not restore power profile: {e}")
                
        if self.watchdog:
            self.watchdog.terminate()
            self.watchdog.wait()

# --- HOW TO USE IT ---
if __name__ == "__main__":
    with LinuxInhibit("Training Alienware Model"):
        print("Starting heavy GPU task...")
        # Put your training loop or VS Code execution here
        # ...