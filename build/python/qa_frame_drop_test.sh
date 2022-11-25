#!/usr/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir="/home/emidan19/gr-tempest/python"
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/emidan19/gr-tempest/build/python":$PATH
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/home/emidan19/gr-tempest/build/swig:$PYTHONPATH
/usr/bin/python3 /home/emidan19/gr-tempest/python/qa_frame_drop.py 
