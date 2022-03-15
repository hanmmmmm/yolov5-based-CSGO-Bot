
import time
import subprocess


def get_active_window():
    
    proc = subprocess.Popen(
        ["xdotool getactivewindow getwindowname"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    return str(out)


if __name__ == "__main__":
    for _ in range(20):
        # os.system( 'xdotool getactivewindow getwindowname' )
        out = get_active_window()
        print("program output:", out )
        
        time.sleep(1)
