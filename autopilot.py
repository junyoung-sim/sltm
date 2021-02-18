#!/usr/bin/env python3

import os, sys
from datetime import datetime, timedelta

end_time   = sys.argv[1] # hh:mm:00
interval   = timedelta(minutes=int(sys.argv[2])) # in minutes

current = str(datetime.now().time())[:8]
while datetime.strptime(current, "%H:%M:%S") - datetime.strptime(end_time, "%H:%M:%S") < timedelta(seconds=0):
    os.system("./scripts/run")
    time.sleep(interval*60)
    current = str(datetime.now().time())[:8]

os.system("clear && ./scripts/run")

