#!/bin/bash
container_name=$1

# create
docker run -itd \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci8 \
--device=/dev/davinci9 \
--device=/dev/davinci10 \
--device=/dev/davinci11 \
--device=/dev/davinci12 \
--device=/dev/davinci13 \
--device=/dev/davinci14 \
--device=/dev/davinci15 \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /data/:/data/ \
-v /home/:/home/ \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /dev/shm:/dev/shm \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--net=host \
--name ${container_name} \
--privileged quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler /bin/bash

# execute
docker exec -it -u root ${container_name} bash