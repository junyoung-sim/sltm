#!/usr/bin/env python3

import os, sys
from datetime import datetime, timedelta

end_time   = sys.argv[1] # hh:mm:00
interval   = timedelta(minutes=int(sys.argv[2])) # in minutes

while str(datetime.now().time())[:8] != end_time:
    os.system("./scripts/run")
    time.sleep(interval*60)

