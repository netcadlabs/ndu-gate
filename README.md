# ndu-gate

Bu proje edge cihazlarda çalışacak ve camera görüntülerinin 
analizi için yüklenen kodları çalıştırmaya yarayan servisi ve 
kullanım senaryolarına özel kodları(runner) ve model verilerini içerir.


## API

### NDUCameraRunner

```api/ndu_camera_runner.py``` dosyasında tanımlı video kaynağından alınan frameleri
 işlemek için gerçeklenecek olan arayüz sınıfıdır.
 
### VideoSource

```api/video_source.py``` dosyasında tanımlı video kaynağı türleri için gerçeklenecek olan arayüz sınıfıdır.

* [PICameraVideoSource](ndu_gate_camera/camera/video_sources/pi_camera_video_source.py)     - Raspberry kamerasından aldığı görüntüyü stream eder.
* [CameraVideoSource](ndu_gate_camera/camera/video_sources/camera_video_source.py)          - İşletim sistemine ait kameradan aldığı görüntüyü stream eder.
* [FileVideoSource](ndu_gate_camera/camera/video_sources/file_video_source.py)              - Ayarlarda verilen klasör ve dosya adını kullanarak video dosyasını stream eder.
* [YoutubeVideoSource](ndu_gate_camera/camera/video_sources/youtube_video_source.py)        - Ayarlarda verilen youtube video linkini stream eder.
* [IPVideoSource](ndu_gate_camera/camera/video_sources/ip_camera_video_source.py)           - TODO

### ResultHandler

```api/result_handler.py``` dosyasında tanımlı runner'lar tarafından üretilen verilerin 
nasıl yönetilmesine karar veren olan arayüz sınıfıdır.

* [ResultHandlerFile](ndu_gate_camera/camera/result_handlers/result_handler_file.py)        - Verileri belirtilen dosyaya yazar
* [ResultHandlerSocket](ndu_gate_camera/camera/result_handlers/result_handler_socket.py)    - Verileri belirtilen socket bağlantısına gönderir
* ResultHandlerRequest  - TODO - Verileri belirtilen servise HTTP(S) ile gönderir
 
##### TODO

## Ayarlar

* ndu-gate isimli servise ait çalışma ayarları  */etc/ndu-gate/config/ndu_gate.yaml* dosyasından değiştirilebilir.

* loglama ayarları */etc/ndu-gate/config/logs.conf* dosyasından değiştirilebilir.


---
 
## Yeni Runner Ekleme

Bu servisin kurulduğu bir cihaza yeni runner eklemek için
 * **/var/lib/ndu_gate/runners/** dizinine **NDUCameraRunner** sınıfından türeyen script(ler) eklenir.
 * **/etc/ndu-gate/config/** dizinine json uzantılı config dosyası eklenir.
 * */etc/ndu-gate/config/ndu_gate.yaml* dosyasında **runners** dizisine ilgili runner ayarları eklenir;
  
```
    runners:
      - 
        name: socialdistance Camera Runner
        type: socialdistance # buradaki deger /var/lib/ndu_gate/runners/ dizininde oluşturulan klasör adı ile aynı olmalıdır.
        configuration: socialdistance.json # Runnera ait özel ayarların bulunduğu ayar dosyası, içerik-format size bağlı
        class: SocialDistanceRunner # Eklenen runner'ın class adı
```


