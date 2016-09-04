import subprocess

f = open('run_log.txt','a+')
while True:
    try:
        output = subprocess.check_output(['th', 'main.lua'])
        f.write('\n')
        f.write(output)
    except:
        pass
f.close()
