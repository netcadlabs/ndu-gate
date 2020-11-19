# FROM python:3.7-slim
FROM jjanzic/docker-python3-opencv:latest

ADD ./ /

RUN echo '#Main start script\n\
CONF_FOLDER="./ndu_gate_camera/config"\n\
firstlaunch=${CONF_FOLDER}/.firstlaunch\n\
\n\
if [ ! -f ${firstlaunch} ]; then\n\
    cp -r /default-config/config/* /ndu_gate_camera/config/\n\
    cp -r /default-config/runners/* /ndu_gate_camera/runners/\n\
    touch ${firstlaunch}\n\
    echo "#Remove this file only if you want to recreate default config files! This will overwrite exesting files" > ${firstlaunch}\n\
fi\n\
echo "nameserver 8.8.8.8" >> /etc/resolv.conf\n\
\n\
python ./ndu_gate_camera/ndu_camera.py\n\
'\
>> start-service.sh && chmod +x start-service.sh

ENV PATH="/root/.local/bin:$PATH"
ENV configs /ndu_gate_camera/config
ENV extensions /ndu_gate_camera/extensions
ENV logs /ndu_gate_camera/logs

RUN python /setup.py install && mkdir -p /default-config/config /default-config/runners/ && cp -r /ndu_gate_camera/config/* /default-config/config/ && cp -r /ndu_gate_camera/runners/* /default-config/runners

RUN echo '#Installation start script\n\
\n\
python3 -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html\n\
\n\
python3 -c "import cv2; print(cv2.__version__)"\n\
\n\
'\
>> installlation.sh && chmod +x start-service.sh

RUN ./installations.sh

# RUN "python3 -m pip install torch torchvision"

VOLUME ["${configs}", "${extensions}", "${logs}"]

CMD [ "/bin/sh", "./start-service.sh" ]
