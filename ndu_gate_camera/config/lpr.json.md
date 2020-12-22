* "rects" verilmezse: 
  * tüm imaj üzerinden plaka aranır.
* verilirse:
  * object detection bu sınıfa ait bir rect bulamadıysa plaka aranmaz
  * bulduysa, her rect için ayrı ayrı aranır. 

"resize": 1200 --> frame veya crop edilen imajın (araç) genişliği 1200'den küçükse, 1200'e resize et.  
defaut değeri "0" --> resize edilmez

Örnek:
```json
{
  "min_confidence_percentage": 80,
  "resize": 1200,
  "rects": [
      "car",
      "bicycle",
      "motorbike",
      "truck",
      "bus"
    ]
}
```

