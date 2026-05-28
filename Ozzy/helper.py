import sys

def bring_window_to_foreground():
    """Bring the MuJoCo viewer window to the foreground (Windows only)"""
    if sys.platform == 'win32':
        import ctypes
        import subprocess
        
        try:
            # Method 1: Try using subprocess to activate the last created window
            # This is more reliable than direct ctypes calls
            subprocess.Popen("powershell -Command \"[Windows.System.Launcher]::LaunchUriAsync('ms-settings:') | Out-Null; Start-Sleep -Milliseconds 100\"", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.1)
            
            # Method 2: Try to find and activate any GLFW window (MuJoCo uses GLFW)
            hwnd = ctypes.windll.user32.FindWindowW(ctypes.c_wchar_p("GLFW30"), None)
            if hwnd:
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                time.sleep(0.1)
                return
            
            # Method 3: Use keyboard shortcut to switch windows (Alt+Tab alternative)
            ctypes.windll.user32.keybd_event(0xA4, 0, 0, 0)  # Alt down
            time.sleep(0.05)
            ctypes.windll.user32.keybd_event(0x09, 0, 0, 0)  # Tab down
            time.sleep(0.05)
            ctypes.windll.user32.keybd_event(0x09, 0, 0x2, 0)  # Tab up
            time.sleep(0.05)
            ctypes.windll.user32.keybd_event(0xA4, 0, 0x2, 0)  # Alt up
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Could not bring window to foreground: {e}")