# ndu-gate

Bu proje edge cihazlarda çalışacak ve camera görüntülerinin 
analizi için yüklenen kodları çalıştırmaya yarayan temel servis ve 
kullanım senaryolarına özel kodları içerir.


# api

## VideoSource

```api/video_source.py``` dosyasında tanımlı video kaynağı türleri için gerçeklenecek olan arayüz sınıfıdır.


## NDUCameraRunner

```api/ndu_camera_runner.py``` dosyasında tanımlı video kaynağından alınan frameleri
 işlemek için gerçeklenecek olan arayüz sınıfıdır.

## ResultHandler

##### TODO


## Yeni Runner Ekleme

Bu servisin kurulduğu bir cihaza yeni runner eklemek için
 * **/var/lib/ndu_gate/runners/** dizinine NDUCameraRunner sınıfından türeyen script(ler) eklenir.
 * 


