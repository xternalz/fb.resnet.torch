import subprocess

while True:
    try:
        subprocess.call(["pkill", "luajit", "-9"])
        output = subprocess.check_output(['th', 'main.lua'])
        f = open('run_log.txt','a+')
        f.write('\n')
        f.write(output)
        f.close()
    except:
        pass
