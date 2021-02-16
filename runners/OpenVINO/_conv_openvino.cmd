::python "C:\Program Files (x86)\Intel\openvino_2021.2.185\deployment_tools\model_optimizer\mo.py" --input_model "C:\_koray\netcadlabs\ndu-gate\runners\yolov4\data\yolov4_-1_3_608_608_dynamic.onnx"

::python "C:\Program Files (x86)\Intel\openvino_2021.2.185\deployment_tools\model_optimizer\mo.py" --input_model "C:\_koray\netcadlabs\ndu-gate\runners\yolov4\data\yolov4-1_3_608_608_static.onnx"
::python "C:\Program Files (x86)\Intel\openvino_2021.2.185\deployment_tools\model_optimizer\mo.py" --input_model "C:\_koray\netcadlabs\ndu-gate\runners\yolov4\data\yolov4-tiny_1_3_416_416_static.onnx"

python "C:\Program Files (x86)\Intel\openvino_2021.2.185\deployment_tools\model_optimizer\mo.py" --input_model "C:\_koray\netcadlabs\ndu-gate\runners\yolov4\data\yolov4-tiny_vehicles_416_static.onnx"
pause