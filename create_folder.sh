#!/bin/bash
NOTE=$1

cd /disk1/thkim/HeteroPFCL
sudo mkdir client_states_${NOTE}
sudo chmod 777 client_states_${NOTE}
cd
cd thkim/HeteroPFCL
ln -s /disk1/thkim/HeteroPFCL/client_states_${NOTE}
