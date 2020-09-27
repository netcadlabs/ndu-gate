# Kurulum hazırlama

AŞağıdaki komut ile *deb_dist/ndu-gate-<versıon>/debian/python3-ndu-gate.deb* klasörüne 
.deb dosyası oluşturulur
```
sudo ./generate_deb_package.sh clean
```

# Kurulum

```
sudo apt install ./python3-ndu-gate.deb -y
```

Kontrol 
```
systemctl status ndu-gate
```

# Problem #1

Aşağıdaki komut sırasında 

```
sudo python3 setup.py --command-packages=stdeb.command bdist_deb
```


```dpkg-checkbuilddeps: error: Unmet build dependencies: dh-python``` şeklinde hata alınırsa; 



* [devscripts](http://manpages.ubuntu.com/manpages/cosmic/man1/mk-build-deps.1.html) ve equivs paketi kurulur.

```
sudo apt-get install dh-python
sudo apt install devscripts
sudo apt install equivs
```

Kurulum çıkartılacak cihazda opencv gibi uygulama içinde olan bağımlılıkların bulunması gerekiyor.