**result_style:** **0**:cumulative(default) - **1**:aggregate 

Örnekler:


``` json
{
  "gates":[[[0.63203125, 0.37777777777777777], [0.05078125, 0.5722222222222222]]],
  "track": "center",
  "result_style" : 1,
  "classes": {
    "arac": [
      "car",
      "bicycle",
      "motorbike",
      "truck",
      "bus"
    ]
  },
  "concat_data_classes": {
      "arac": ["PL:*"]
  }
}

{
  "gates":[[[0.63203125, 0.37777777777777777], [0.05078125, 0.5722222222222222]]],
  "track": "center",
  "result_style" : 1,
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
    ]
  }
}

{
  "gates(canakkale_cam1)":[[[0.8736979166666666, 0.2824074074074074], [0.6393229166666666, 0.2569444444444444]]],
  "gates(vid_short)":[[[0.86796875, 0.6916666666666667], [0.1359375, 0.48194444444444445]], [[0.50703125, 0.7444444444444445], [0.33515625, 0.675]], [[0.39375, 0.23194444444444445], [0.515625, 0.275]]],
  "track": "center",
  "classes": {
    "kisi": [
      "person"
    ],
    "arac": [
      "car",
      "bicycle",
      "motorbike",
      "truck",
      "bus"
    ]
  }
}
```