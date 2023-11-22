import os
import time

def GetLocalTime(strStyle = '%04d-%02d-%02d-%02d-%02d-%02d'):
    tm = time.localtime()
    return (strStyle %(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec))

class MyLogger:
    def __init__(self, args):
        self.f = open(os.path.join(args.save_dir, ("log-%s.txt" %GetLocalTime())), 'a')
    
    def writeLog(self, string, printF = True):
        print(string)
        self.f.write(string + '\n')
        self.f.flush()
    
    def close(self, nextLineF = True):
        if (nextLineF):
            self.f.write('\n')
        self.f.close()
    
    def __del__(self):
        self.f.close()
        print('MyLogger destructor called')