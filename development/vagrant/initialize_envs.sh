#!/bin/bash
source /home/vagrant/.profile
mkvirtualenv -p /usr/bin/python3 $WORKON_HOME/robustToolbox
cd /home/vagrant/robustToolbox/package
workon robustToolbox
cat requirements.txt | xargs -n 1 pip install

