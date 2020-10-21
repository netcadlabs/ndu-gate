# Runner Listesi

Runnerların kullandığı modeller github'a eklenmemiştir. 
Onları şuradan indirebilirsiniz: https://drive.google.com/drive/u/1/folders/1JU5SIONJypgZAFJHesOaGCGsdRDrIume

Çalışması istenen runnerlar için **ndu_gate.yaml** içerisine tanımlarının eklenmesi gerekir.

Örnek:
```yaml
runners:
  - name: Face Detector
    type: facedetector # ndu-gate/runners dizini altındaki runnerın bunuduğu klasörün adı.
    configuration: facedetector.json # ndu-gate/ndu_gate_camera/config dizinde bulunan, runner'a ait özel ayarların yapılabildiği congif dosyası.
    class: face_detector_runner # çalıştırılacak runner class'ının adı.
    priority: 10  # Öncelik sırası
```

* ***priority*** değeri düşük olak runner'lar daha önce çalıştırılır.
* Bir runner kendisinden önce çalışmış başka bir runner'ın çıktısını kullanabilir.
    * Bir runnner'ın process_frame sonucu list [] olmalıdır. Liste içindeki her eleman için 
      *ndu-gate/ndu_gate_camera/utility/constants.py* 'daki RESULT_KEY_* değerleri kullanılabilir. 
        ```python    
        # [x1, y1, x2, y2]
        RESULT_KEY_RECT = "rect"
        # string
        RESULT_KEY_CLASS_NAME = "class_name"
        # 0-1 arasında float olasılık değeri.
        RESULT_KEY_SCORE = "score"
        # Platform'a gönderilmesi istenen veri. İçeriği dictionary olmalıdır.
        RESULT_KEY_DATA = "data"
        ```
 
## Priority = 10 Runners
### Coco 80 - Object Detection - Runners:
Coco veriseti ile eğitilmiş 80 sınıf tanıyan modellerdir. Rect, Class_Name ve Score dönerler.
Data dönmezler, yani platforma doğrudan bilgi göndermezler. Tanınan nesneler:
```
person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa, potted plant, bed, dining table, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```
Performans ölçümü https://youtu.be/N9JtUC8hl8o videosunun ilk sahnesi kullanılarak Macbook'ta yapılmıştır
#### yolov3
En yüksek bulma başarısına sahip modeldir. ***210 msec***

Not: 
#### ssd_mobilenet
İyi bulma başarısına sahip, performansı yüksek bir modeldir. ***38 msec***
#### yolov3-tiny
En düşük bulma başarısına sahip, en hızlı çalışan modeldir.  ***25 msec***

Not: yolo-darknet'in geliştiricisi v3 sonrasında computer vision işlerinden çekilmeye karar vermiş. 
Yaptığım denemelerde yolov4 onnx daha başarılı sonuçlar vermedi. Bu yüzden 
yolov4 ve v5 olduğu söyleyenen modelleri eklemedik veya denemedik.
### Face Detection
#### facedetector
Yüz rectangle ve score bulur. Config dosyasından toleransı ayarlanabilir. ***33 msec***


## Priority = 100 Runners
### personcounter
Kişileri sayar. Dummy bir runner'dır, gelen result içersinden class_name "person" olan değerleri sayar.
"person" class_name dönen herhangi bir düşük priority değerine sahip runner ile birlikte çalıştırılmalıdır. 
**ssd_mobilenet** runner'ı ile optimum başarı/performans ile çalışıyor. 

### drivermonitor
Birlikte çalıştığı runnerlar: yolov3, facedetector
Telefonla konuşma: face detector'un bulduğu "face" dikdörtgenlerine dokunan, yolov3'ün bulduğu 
"cell phone" dikdörtgeni varsa telefonla konuşuluyor kararı alır.
Emniyet kemeri: yolov3'ün bulduğu "person" dikdörgenini crop eder ve googlenet-9 classification
modeli ile emniyet kemeri bulmaya çalışır.

### emotion
"face" rectangle dönen bir runner ile çalışır. (facedetector)
Çıktısı face rectangle ve o face için bulunan duygu adıdır. Duygular: 
```
neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
```
