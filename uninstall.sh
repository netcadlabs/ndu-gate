SERVICE_NAME=ndu-gate
systemctl stop $SERVICE_NAME
systemctl disable $SERVICE_NAME
rm /etc/systemd/system/$SERVICE_NAME
rm /etc/systemd/system/$SERVICE_NAME # and symlinks that might be related
rm /usr/lib/systemd/system/$SERVICE_NAME
rm /usr/lib/systemd/system/$SERVICE_NAME # and symlinks that might be related
systemctl daemon-reload
systemctl reset-failed

sudo apt remove python3-ndu-gate

pip3 uninstall python3-ndu-gate