
CURRENT_VERSION=$( grep -Po 'VERSION[ ,]=[ ,]\"\K(([0-9])+(\.){0,1})+' setup.py )
if [ "$1" = "clean" ] || [ "$1" = "only_clean" ] ; then
  sudo rm -rf for_build/etc/ndu-gate/*
  sudo rm -rf deb_dist/
  sudo rm -rf dist/
  sudo rm -rf ndu_gate.egg-info/
  sudo rm -rf /etc/ndu-gate/
  sudo rm -rf ndu-gate-$CURRENT_VERSION.tar.gz
  sudo rm -rf python3-ndu-gate.deb
  sudo apt remove python3-ndu-gate -y
fi

if [ "$1" != "only_clean" ] ; then
  echo "Installing libraries for building deb package."
  sudo apt-get install python3-stdeb fakeroot python-all
  echo "Building DEB package"
  echo "Creating sources for DEB package..."
  python3 setup.py --command-packages=stdeb.command bdist_deb
  echo "Adding the files, scripts and permissions in the package"
  sudo cp -r ndu_gate_camera/runners for_build/etc/ndu-gate/runners
  sudo cp -r ndu_gate_camera/config for_build/etc/ndu-gate/config
  sudo cp -r ndu_gate_camera/data for_build/etc/ndu-gate/data
  sudo cp -r for_build/etc deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate
  sudo cp -r for_build/var deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate
  sudo cp -r -a for_build/DEBIAN deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate
  sudo chown root:root deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/ -R
  sudo chown root:root deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/var/ -R
  sudo chmod 775 deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/DEBIAN/preinst
  sudo chmod +x deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/DEBIAN/postinst
  sudo chown root:root deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/DEBIAN/preinst
  sudo sed -i '/^Depends: .*/ s/$/, libffi-dev, libglib2.0-dev, libxml2-dev, libxslt-dev, libssl-dev, zlib1g-dev/' deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/DEBIAN/control >> deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/DEBIAN/control
  # Bulding Deb package
  dpkg-deb -b deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate/
  cp deb_dist/ndu-gate-$CURRENT_VERSION/debian/python3-ndu-gate.deb .
fi
