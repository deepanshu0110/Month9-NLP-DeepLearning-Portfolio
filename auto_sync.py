import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── CONFIG ────────────────────────────────────────────────────────────────
WATCH_DIR   = r"C:\Users\Deepanshu\OneDrive\Desktop\Month9"
REPO_DIR    = r"C:\Users\Deepanshu\OneDrive\Desktop\Month9"
EXTENSIONS  = {'.ipynb', '.py', '.png', '.csv', '.txt', '.md', '.json', '.h5', '.keras'}
# ─────────────────────────────────────────────────────────────────────────

class SyncHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_push = 0
        self.cooldown  = 15   # seconds between pushes

    def should_track(self, path):
        return any(path.endswith(ext) for ext in EXTENSIONS)

    def push(self, event_path):
        now = time.time()
        if now - self.last_push < self.cooldown:
            return
        self.last_push = now

        fname = os.path.basename(event_path)
        print(f"\n[auto_sync] Change detected: {fname}")

        try:
            subprocess.run(['git', 'add', '.'],             cwd=REPO_DIR, check=True)
            subprocess.run(['git', 'commit', '-m',
                            f'auto-sync: {fname}'],         cwd=REPO_DIR, check=True)
            subprocess.run(['git', 'push', 'origin', 'main'], cwd=REPO_DIR, check=True)
            print(f"[auto_sync] Pushed: {fname}")
        except subprocess.CalledProcessError as e:
            print(f"[auto_sync] Git error: {e}")

    def on_modified(self, event):
        if not event.is_directory and self.should_track(event.src_path):
            self.push(event.src_path)

    def on_created(self, event):
        if not event.is_directory and self.should_track(event.src_path):
            self.push(event.src_path)


if __name__ == '__main__':
    print(f"[auto_sync] Watching: {WATCH_DIR}")
    print(f"[auto_sync] Tracking: {EXTENSIONS}")
    print("[auto_sync] Press Ctrl+C to stop.\n")

    handler  = SyncHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
