from subprocess import Popen, PIPE
import subprocess

f = open('run_log.txt','a+')
while True:
    try:
        subprocess.call(["pkill", "luajit", "-9"])
        p = Popen(['th', 'main.lua', '-retrain'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        f.write('\n')
        f.write(output)
        f.write(err)
    except:
        pass
f.close()
