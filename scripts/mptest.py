import itertools as it
from multiprocessing import Process, Pipe

class MockPH:
    def __init__(self):
        pass
    def max_load(self):
        return 100
    def angs(self):
        return {'a':1, 'b':2}
    def goto_position(self, angs):
        print('moved to ' + str(angs))
        return None
    def close(self):
        print('closed')
        return None

def ticker(period, conn):
    # periodically print motor load
    # execute commands as received
    ph = MockPH()
    for i in it.count():
        print(i, ph.max_load(), "\n>>>")
        for t in range(period):
            if conn.poll():
                name, args = conn.recv()
                if name == "q":
                    ph.close()
                    return
                getattr(ph, name)(*args)

class Controller:
    def __init__(self, period):
        self.pipe, receiver = Pipe()
        self.ticker = Process(target=ticker, args=(period, receiver))
        self.ticker.start()
    def quit(self):
        self.pipe.send(("q", ()))
        self.ticker.join()
    def goto_position(self, angs):
        self.pipe.send(("goto_position", (angs,)))

period = 10**6
print(MockPH().__dict__)

ctr = Controller(period)
ctr.goto_position(None)

# parent.send(("goto_position", (None,)))
# parent.send(("q", ()))

