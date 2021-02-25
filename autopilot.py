#!/usr/bin/env python3

import os, sys, time
from datetime import datetime, timedelta

end_time = sys.argv[1] # hh:mm:00
interval = int(sys.argv[2]) # in minutes

itr = 0
current = str(datetime.now().time())[:8]
while datetime.strptime(current, "%H:%M:%S") - datetime.strptime(end_time, "%H:%M:%S") < timedelta(seconds=0):
    os.system("./scripts/run")
    print("\n\n\nWAITING FOR NEXT INTERVAL... {}\n\n\n" .format(itr))
    time.sleep(interval*60)
    current = str(datetime.now().time())[:8]
    itr += 1

os.system("clear && ./scripts/run")
