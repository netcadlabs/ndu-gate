# ndu-gate

Bu proje edge cihazlarda çalışacak ve camera görüntülerinin 
analizi için yüklenen kodları çalıştırmaya yarayan servisi ve 
kullanım senaryolarına özel kodları(runner) ve model verilerini içerir.


## API

### VideoSource

```api/video_source.py``` dosyasında tanımlı video kaynağı türleri için gerçeklenecek olan arayüz sınıfıdır.


### NDUCameraRunner

```api/ndu_camera_runner.py``` dosyasında tanımlı video kaynağından alınan frameleri
 işlemek için gerçeklenecek olan arayüz sınıfıdır.

### ResultHandler

##### TODO

## Ayarlar

* ndu-gate isimli servise ait çalışma ayarları  */etc/ndu-gate/config/ndu_gate.yaml* dosyasından değiştirilebilir.

* loglama ayarları */etc/ndu-gate/config/logs.conf* dosyasından değiştirilebilir.

## Yeni Runner Ekleme

Bu servisin kurulduğu bir cihaza yeni runner eklemek için
 * **/var/lib/ndu_gate/runners/** dizinine **NDUCameraRunner** sınıfından türeyen script(ler) eklenir.
 * **/etc/ndu-gate/config/** dizinine json uzantılı config dosyası eklenir.
 * */etc/ndu-gate/config/ndu_gate.yaml* dosyasında **runners** dizisine ilgili runner ayarları eklenir;
    * ```
        runners:
          - name: socialdistance Camera Runner
            type: socialdistance # buradaki deger /var/lib/ndu_gate/runners/ dizininde oluşturulan klasör adı ile aynı olmalıdır.
            configuration: socialdistance.json # Runnera ait özel ayarların bulunduğu ayar dosyası, içerik-format size bağlı
            class: SocialDistanceRunner # Eklenen runner'ın class adı
        ```


