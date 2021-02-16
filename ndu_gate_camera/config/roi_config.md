ndu_gate yaml dosyasına tanımlanan 'runner'lara "roi_config" eklenebilir. Örnek:

      - name: yolov4 - Object detection with 80 classes
        type: yolov4
        configuration: yolov4.json
        class: Yolov4Runner
        roi_configuration: roi_config_example.json
        priority: 10

Bu json diğer runner config json dosyalarının da olduğu config dizininde olmalıdır.

Örnek içerik:
    
    {
      "polygons": [[[619, 330], [773, 339], [769, 213], [637, 214]]],
      "apply_mask": true,
      "apply_crop": true,
      "preview": true,
      "pyrUp": 2
    }

* **polygons**: Maske yapılacak veya crop için kullanılacak poligonlar. Sadece bu poligonların içi için işlem yapılacaktır.
* **apply_mask**: Maske yapılsın mı? *Varsayılan değer: true*
* **apply_crop**: Poligonlara göre imaj kesilsin mi? Poligonların hepsinin min-max koordinatlarına göre oluşturulan bbox kullanılarak imaj kesilir ve küçültülmüş bu imaj üzerinden işlem yapılır. *Varsayılan değer: true* 
* **preview**: Debug amaçlıdır. Önizleme yapmak için kullanılır. Varsayılan değer: false
* **pyrUp**: Piramit büyütme kaç kere yapılsın? Varsayılan değer: 1 (büyütme yapılmasın)