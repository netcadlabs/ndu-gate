# ndu-gate - NDU Gate Camera Service

This is a service project that run on edge devices and servers to consume videos sources(camera, file etc.) and process them.

## Installation

### Using .deb package

Go relases and download latest .deb package and run the following command.

```
sudo dpkg -i ./python3-ndu-gate.deb
```

Building .deb package

```
sudo ./generate_deb_package.sh
```

### Using pip package
```
python3 -m pip install --upgrade ndu-gate-camera
```


* installation without dependencies


```
python3 -m pip install --upgrade --no-deps ndu-gate-camera
```

### Running with cli

```
ndu-gate -c config-file-path.yaml
```

--- 
## API

### NDUCameraRunner

It is a interface class(```api/ndu_camera_runner.py```) that sould be impelemented to process source frames according to different use-cases.
 
### VideoSource

It is a interface class (```api/video_source.py```) that should be implement to stream different type of video sources.


* [CameraVideoSource](ndu_gate_camera/camera/video_sources/camera_video_source.py)          - Streams frames from OS camera
* [FileVideoSource](ndu_gate_camera/camera/video_sources/file_video_source.py)              - Streams frames from video file.
* [ImageVideoSource](ndu_gate_camera/camera/video_sources/image_video_source.py)              - Streams frames from image file.
* [YoutubeVideoSource](ndu_gate_camera/camera/video_sources/youtube_video_source.py)        - Streams frames from youtube video url.
* [IPVideoSource](ndu_gate_camera/camera/video_sources/ip_camera_video_source.py)           - Streams frames from IP camera
* [PICameraVideoSource](ndu_gate_camera/camera/video_sources/pi_camera_video_source.py)     - Streams frames from Raspberry PI camera

### ResultHandler

 It is the interface class (```api/result_handler.py```) that decides how to manage the data produced by runners.

* [ResultHandlerFile](ndu_gate_camera/camera/result_handlers/result_handler_file.py)        - Writes the data to the specified file
* [ResultHandlerSocket](ndu_gate_camera/camera/result_handlers/result_handler_socket.py)    - Sends the data to the specified socket connection
* ResultHandlerRequest  - TODO - Sends data to the specified service via HTTP(S)


## Settings

* ndu-gate service global settings */etc/ndu-gate/config/ndu_gate.yaml*

* Logging settings : */etc/ndu-gate/config/logs.conf*


---
 
## Adding New Runner

You can a new implemented runner to this service. 

 * Create a folder under **/var/lib/ndu_gate/runners/**. This folder name should be unique.
 * Add your **NDUCameraRunner** implementation python file to **/var/lib/ndu_gate/runners/** folder.
 * Add your runner's config file to **/etc/ndu-gate/config/<folder-name>**
 * Then to activiate your runner, add the following settings top under instance runners collection in */etc/ndu-gate/config/ndu_gate.yaml* file
  
```
    instance:
      - source
        type: CAMERA
        device: MyLapCamera # optional
        runners:
          - name: My Runner
            type:  # this should be same with <folder-name>
            configuration: <folder-name>.json # optional
            class: MyRunner # The class name your runner class
```


