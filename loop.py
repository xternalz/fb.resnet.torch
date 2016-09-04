import subprocess

f = open('run_log.txt','a+')
while True:
    try:
        subprocess.call(["pkill", "luajit", "-9"])
        output = subprocess.check_output(['th main.lua -retrain'], stderr=subprocess.STDOUT, shell=True)
        f.write('\n')
        f.write(output)
    except:
        pass
f.close()
