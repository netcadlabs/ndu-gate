

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