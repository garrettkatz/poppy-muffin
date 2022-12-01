import signal

done = False

def handler(signum, frame):
    global done
    print('here')
    done = True
signal.signal(signal.SIGINT, handler)

def busy():
    global done
    done = False
    while not done: pass
    print('done')

print('start')
busy()
print('start again')
busy()
