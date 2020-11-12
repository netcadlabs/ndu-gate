# Intersector

## intersector.json
```yaml
{
  "groups": {
    "<group_name1>": { //n tane grup olabilir
      "obj_detection": { //0 veya 1 tane obj_detection olabilir. Önceki çalıştırılmış object detectorlardan gelen sonuçlar yorumlanır.

        //ground ve dist opsiyoneldir. ground veya dist yoksa hesaplamalar 2D bbox'a göre yapılır.
        "ground": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], //rect'ler arasındaki mesafenin kuş bakışı hesaplanması için kullanılan zemin üzerindeki bir karedir.
        "dist": [[x1,y1],[x2,y2]], //iki rect'in birbirine kuş bakışı bu mesafe kadar yakınlığı hesaplanır. Sadece rect'in "style" değeri "dist" olarak tanımlananlar için kullanılır.

        "rects": [
           {//n tane rect olabilir. Birden fazla olduğunda AND şeklinde çalışır; yani tüm rect elemanlarının koşullanının sağlanmış olması gerekir.
             
             "padding": 0, //Default=0. rect'lere padding vermek istediğimizde kullanılır. Örneğin 0.1 -> %10 büyüt demektir.
             
             "style": "<or, and, touch, dist>", //Default="or" 
                    //or: class_names listesindeki isme sahip herhangi bir rectangle var mı?
                    //and: class_names listesindeki isimlerin hepsi var mı? 
                    //touch: class_names listesindeki rect elemanları birbirlerine 2D dokunuyorlar mı? Varsa padding dikkate alınır.
                    //dist: kuş bakışı görüntüde dist uzunluğundan birbirine daha yakın rect var mı?
             
             //önceki çalışmış object detection runnerlarının bulduğu rect isimleri. 
             //'*' karakteri kullanılabilir.
             //aynı class_name birden fazla kullanılabilir. Örneğin sosyal mesafe ihlali için 2 tane "person" eklenebilir.
             "class_names": ["<class_name1>", "<class_name2>", ...]   
           },
           ...
        ]
      },
      "classification": { //0 veya 1 tane classification olarbilir.        
        "threshold": 0.1, //classification minimum score değeri. Default=0.5
        "padding": 0.03, //classification çalıştırılacak rect'lere padding uygulamak istendiğinde kullanılır. Default=0
        "rects": ["<class_name1>","<class_name1>", ...], //bu isimlerdeki tüm rect'ler için classification çalıştırılır. '*' karakteri kullanılabilir.
                                                         //hiç rect verilmemişse frame'in tamamına classification yapılır.
        "classify_names": ["<class_name1>", "<class_name2>", ...] //bu isimlere ait bir classification sonucu bulundu mu? 
                                                          //Her zaman 'or' olarak çalışır. Yani bir tanesi varsa kabul edilir.
                                                          //yani rectlerden herhangi birisinden listedeki herhangi bir classify namelerden birisi bulunursa kabul edilir.        
      }
      //hem "obj_detection" hem de "classification" tanımı yapılmışsa "or" olarak çalışır. Önce obj_detection çalışır, true çıkmışsa, classification çalıştırılmaz.
    },
    ...
  }
}
```


### Örnek:
```json
{
  "groups": {
    "Sosyal mesafe ihlali": {
      "obj_detection": {
        "ground": [[1,1],[2,2],[3,3],[4,4]],
        "dist": [[1,1],[2,2]],
        "rects": [
          {
            "padding": 0,
            "style": "dist",
            "class_names": [
              "person",
              "person"
            ]
          }
        ]
      }
    },
    "Is guvenligi ihlali": {
      "obj_detection": {
        "ground": [0,1,2,3],
        "dist": [[0,1],[2,3]],
        "rects": [
          {
            "padding": 0,
            "class_names": [
              "person"
            ]
          },
          {
            "padding": 0,
            "style": "or",
            "class_names": [
              "truck",
              "bus"
            ]
          }
        ]
      }
    },
    "Sigara iciyor": {
      "obj_detection": {
        "rects": [
          {
            "padding": 0,
            "style": "touch",
            "class_names": [
              "cigarette",
              "face"
            ]
          }
        ]
      }
    },
    "Emniyet kemeri takili": {
      "classification": {
        "threshold": 0.1,
        "padding": 0.03,
        "rects": [
          "person"
        ],
        "classify_names": [
          "*seatbelt*"
        ]
      }
    },
    "Telefonla konusuyor": {
      "obj_detection": {
        "rects": [
          {
            "style": "touch",
            "padding": 0.001,
            "class_names": [
              "cell phone",
              "face"
            ]
          }
        ]
      },
      "classification": {
        "threshold": 0.1,
        "padding": 0,
        "rects": [
          "face"
        ],
        "classify_names": [
          "*phone*"
        ]
      }
    },
    "Kamera tasit tespit etti!": {
      "obj_detection": {
        "rects": [
          {
            "style": "or",
            "class_names": [
              "car",
              "bicycle",
              "motorbike",
              "truck",
              "bus"
            ]
          }
        ]
      }
    },
    "Ayi saldırma tehlikesi": {
      "obj_detection": {
        "rects": [
          {
            "style": "and",
            "class_names": [
              "person",
              "bear"
            ]
          }
        ]
      }
    },
    "Vahsi hayvan saldırma tehlikesi": {
      "obj_detection": {
        "rects": [
          {
            "class_names": [
              "person"
            ]
          },
          {
            "style": "or",
            "class_names": [
              "elephant",
              "bear",
              "zebra",
              "giraffe"
            ]
          }
        ]
      }
    }
  }
}


