### tracker_algorithm değerleri
```javascript
  "tracker_algorithm": "KCF_600"    # fps:35  - Default  - plaka tanıma işlerini bozuyor, rect yaklaşan araç için büyümüyor
  "tracker_algorithm": "KCF"        # fps:17  -
  "tracker_algorithm": "CSRT"       # fps:4   - Başarısı daha iyi ama yavaş
  "tracker_algorithm": "MOSSE"      # fps:150 - Başarı düşük ama çok hızlı - plaka tanma işi için KCF'den daha iyi (rect boyutu daha iyi korundu)

  "tracker_algorithm": "Boosting"   # fps:4
  "tracker_algorithm": "MIL"        # fps:2
  "tracker_algorithm": "TLD"        # fps:1
  "tracker_algorithm": "MedianFlow" # fps:15
  "tracker_algorithm": "GOTURN"     # fps:3
```

Örnek:
```json
{
  "tracker_algorithm": "KCF_600",
  "classes": {
    "arac": [
      "car",
      "bicycle",
      "motorbike",
      "truck",
      "bus"
    ],
    "kisi": [
      "person"
    ],
    "plaka": [
      "PL:*"
    ]
  }
}
```