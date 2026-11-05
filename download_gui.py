import tkinter as tk
from tkinter import messagebox
import subprocess
import psutil
import threading
import sys
import os

class DownloadManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Downloader")
        self.root.geometry("350x150")
        
        self.process = None
        self.ps_process = None
        
        self.lbl_status = tk.Label(root, text="Ready to download", font=("Arial", 10))
        self.lbl_status.pack(pady=10)
        
        self.btn_start = tk.Button(root, text="Start Download", command=self.start_download, width=15)
        self.btn_start.pack(pady=5)
        
        frame = tk.Frame(root)
        frame.pack(pady=5)
        
        self.btn_pause = tk.Button(frame, text="Pause", command=self.pause_download, state=tk.DISABLED, width=10)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_resume = tk.Button(frame, text="Resume", command=self.resume_download, state=tk.DISABLED, width=10)
        self.btn_resume.pack(side=tk.LEFT, padx=5)
        
    def start_download(self):
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.lbl_status.config(text="Downloading... (Check console for progress bar)")
        
        # Start download in a separate thread so GUI doesn't freeze
        threading.Thread(target=self.run_download_script, daemon=True).start()
        
    def run_download_script(self):
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_dataset.py")
        
        # Run the existing download_dataset.py script
        self.process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=sys.stdout,  # Pipe output directly to the console so tqdm works perfectly
            stderr=sys.stderr
        )
        
        try:
            self.ps_process = psutil.Process(self.process.pid)
        except psutil.NoSuchProcess:
            pass
            
        self.process.wait()
        self.root.after(0, self.download_finished)
        
    def pause_download(self):
        if self.ps_process and self.ps_process.is_running():
            self.ps_process.suspend() # Suspends the OS process
            self.lbl_status.config(text="Paused")
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.NORMAL)
            print("\n[Download Paused]")
            
    def resume_download(self):
        if self.ps_process and self.ps_process.is_running():
            self.ps_process.resume() # Resumes the OS process
            self.lbl_status.config(text="Downloading... (Check console for progress bar)")
            self.btn_resume.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            print("\n[Download Resumed]")
            
    def download_finished(self):
        self.lbl_status.config(text="Download Finished!")
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.NORMAL)
        if self.process and self.process.returncode == 0:
            messagebox.showinfo("Success", "Dataset download completed.")
        else:
            messagebox.showerror("Error", "Download failed or was interrupted.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DownloadManager(root)
    root.mainloop()
