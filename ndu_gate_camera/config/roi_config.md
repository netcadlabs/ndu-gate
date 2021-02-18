ndu_gate yaml dosyasına tanımlanan 'runner'lara *roi_config* eklenebilir. Örnek:

      - name: yolov4 - Object detection with 80 classes
        type: yolov4
        configuration: yolov4.json
        class: Yolov4Runner
        roi_configuration: roi_config_example.json
        priority: 10

Bu json diğer runner config json dosyalarının da olduğu config dizininde olmalıdır.

Örnek içerik:
    
    {
      "select_polygons_mode": false,
      "polygons": [[[619, 330], [773, 339], [769, 213], [637, 214]]],
      "apply_mask": true,
      "apply_crop": true,
      "preview": true,
      "pyrUp": 2
    }

* **select_polygons_mode** *Default: false* json dosyasına polygons kısmına ne yazacağımızı kolaylaştırmak için kullanılır.
  Poligonları çizebileceğimiz bir ekran açılır ve çizim sonrasında console'a polygons içeriği yazılır ve program sonlandırılır.
  "**n**" ile bir sonraki poligon çizilebilir.
  "**s**" ile çizim sonlandırılır.
* **polygons**: Maske yapılacak veya crop için kullanılacak poligonlar. Sadece bu poligonların içi için işlem yapılacaktır.
* **apply_mask**: Maske yapılsın mı? *Varsayılan değer: true*
* **apply_crop**: Poligonlara göre imaj kesilsin mi? Poligonların hepsinin min-max koordinatlarına göre oluşturulan bbox kullanılarak imaj kesilir ve küçültülmüş bu imaj üzerinden işlem yapılır. *Varsayılan değer: true* 
* **preview**: Debug amaçlıdır. Önizleme yapmak için kullanılır. Varsayılan değer: false
* **pyrUp**: Piramit büyütme kaç kere yapılsın? Varsayılan değer: 1 (büyütme yapılmasın)

Poligon koordinatlarını kolay doldurmak için:
* *polygons* içerisi boş olsun []
* program çalışınca ilk frame üzerinden poligon seçtirme ekranı açılacak. 
  Birden fazla poligon için "n" tuşuna basın. Çizimi bitirmek için "s" tuşuna basın.
  Consola polygons içeriği print edilecek. Bunu koplayatıp roi_config dosyasının içine yapıştırabilirsiniz.