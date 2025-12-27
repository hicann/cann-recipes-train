# Copyright 2025 Chinese Information Processing Laboratory, ISCAS.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM verl:main-c651b7b-py311-cann8.3.RC1

############################ base ############################

COPY ./ /root/sandbox

ENV DEBIAN_FRONTEND=noninteractive
RUN bash /root/sandbox/scripts/tuna-apt-arm64.sh 22.04 \
    && apt-get update && apt-get install -y curl npm git nano wget vim unzip sudo cgroup-tools iproute2 iptables \
    # iverilog build deps
    autoconf gperf flex bison \
    # bash scripting utils
    bc \
    && mkdir -p /workspace/download

# python 3.11 & poetry
# use new base environment to avoid conflict
RUN . /home/ma-user/miniconda3/bin/activate && conda create -n sandbox-base python=3.11 -y
ENV PATH="/home/ma-user/miniconda3/envs/sandbox-base/bin:${PATH}"
RUN wget https://veml.tos-cn-beijing.volces.com/condarc -O ~/.condarc && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.7.0 python3 -
ENV PATH=/root/.local/bin:$PATH

# 1. golang 1.23.3 (arm64)
RUN curl -o /workspace/sandbox/download/go.tar.gz -SL https://golang.google.cn/dl/go1.23.3.linux-arm64.tar.gz \
    && tar -zxf /workspace/sandbox/download/go.tar.gz -C /usr/local && rm /workspace/sandbox/download/go.tar.gz
ENV PATH=/bin:/usr/local/go/bin:$PATH

# 2. nodejs 20.11.0 (arm64)
RUN curl -o /workspace/download/node.tar.gz -SL https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-arm64.tar.gz \
    && mkdir -p /usr/local/lib/nodejs && tar -zxf /workspace/download/node.tar.gz -C /usr/local/lib/nodejs && mv /usr/local/lib/nodejs/node-v20.11.0-linux-arm64 /usr/local/lib/nodejs/node \
    && rm /workspace/download/node.tar.gz
ENV PATH=/usr/local/lib/nodejs/node/bin:$PATH
ENV NODE_PATH=/usr/local/lib/node_modules
RUN npm install -g typescript@5.3.3 tsx@4.7.1

# 3. gcc 9
RUN apt-get update && apt-get install -y build-essential g++ libboost-all-dev

# 4. OpenSSL (arm64)
RUN curl -o /workspace/download/openssl.tar.gz -SL https://www.openssl.org/source/old/3.0/openssl-3.0.11.tar.gz \
    && tar -zxf /workspace/download/openssl.tar.gz && cd openssl-3.0.11 && ./Configure linux-aarch64 && make && make install \
    && rm /workspace/download/openssl.tar.gz && cd .. && rm -r openssl-3.0.11
ENV PATH=/usr/bin/openssl:$PATH

# 5. jdk 21 (arm64)
RUN curl -o /workspace/download/jdk.tar.gz -SL https://download.oracle.com/java/21/latest/jdk-21_linux-aarch64_bin.tar.gz \
    && mkdir /usr/java && tar -zxf /workspace/download/jdk.tar.gz -C /usr/java && rm /workspace/download/jdk.tar.gz \
    && mv /usr/java/jdk-21* /usr/java/jdk-21
ENV JAVA_HOME=/usr/java/jdk-21
ENV PATH=$JAVA_HOME/bin:$PATH

# 6. dotnet 8.0 (arm64)
RUN apt-get update \
    && apt-get install -y wget tar \
    && curl -L -o /workspace/download/dotnet-install.sh https://dot.net/v1/dotnet-install.sh \
    && chmod +x /workspace/download/dotnet-install.sh \
    && /workspace/download/dotnet-install.sh --install-dir /usr/share/dotnet --channel 8.0 --version latest \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm /workspace/download/dotnet-install.sh

# 7. php 8.1
RUN apt-get install -y php8.1-cli

# 8. rust (arm64)
ENV RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
ENV RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain 1.76.0 --profile minimal --target aarch64-unknown-linux-gnu
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

# 9. R, lua, ruby, julia, perl, scala (ubuntu:22.04)
RUN apt-get install -y r-base ruby-full scala \
    && PERL_MM_USE_DEFAULT=1 cpan Test::Deep Data::Compare

# 10. D
RUN apt install -y clang lld \
    && cd /workspace/download \
    && tar xf ldc2-1.38.0-linux-aarch64.tar.xz \
    && sudo mv ldc2-1.38.0-linux-aarch64 /opt/ldc \
    && echo 'export PATH=/opt/ldc/bin:$PATH' | sudo tee /etc/profile.d/ldc.sh \
    && . /etc/profile.d/ldc.sh \
    && ldc2 --version

# 11. kotlin (JVM-based)
RUN curl -L -o /workspace/download/kotlin-compiler.zip https://github.com/JetBrains/kotlin/releases/download/v2.0.0/kotlin-compiler-2.0.0.zip \
    && mkdir -p /usr/local/kotlin \
    && unzip /workspace/download/kotlin-compiler.zip -d /usr/local/kotlin \
    && rm -f /workspace/download/kotlin-compiler.zip \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/kotlin/kotlinc/bin:$PATH

# 12. Racket (arm64)
RUN apt-get update --allow-insecure-repositories -y && apt-get install -y racket

# 13. Swift (arm64)
RUN curl -o /workspace/download/swift.tar.gz -SL https://download.swift.org/swift-5.10.1-release/ubuntu2004-aarch64/swift-5.10.1-RELEASE/swift-5.10.1-RELEASE-ubuntu20.04-aarch64.tar.gz \
    && cd /workspace/download \
    && tar zxf swift.tar.gz \
    && mkdir -p /usr/local/swift \
    && mv swift-5.10.1-RELEASE-ubuntu20.04-aarch64/usr/* /usr/local/swift \
    && rm -f /workspace/download/swift.tar.gz
ENV PATH=/usr/local/swift/bin:$PATH

# clean
RUN update-alternatives --install /usr/bin/java java $JAVA_HOME/bin/java 20000 \
    && update-alternatives --install /usr/bin/javac javac $JAVA_HOME/bin/javac 20000 \
    && aarch64-linux-gnu-gcc --version \
    && dotnet --info | grep RID \
    && go version | grep arm64 \
    && node -p "process.arch" | grep arm64 \
    && java -version \
    && rustc -Vv | grep host | grep aarch64

############################ server ############################

# download and cache go deps
RUN cd /root/sandbox/runtime/go \
    && go env -w GOPROXY="https://goproxy.cn|direct" \
    && go build

# also install puppetter chrome requirements (TODO: keep effective packages only)
RUN cd /root/sandbox/runtime/node && \
    npm config set registry https://registry.npmmirror.com && \
    npm ci
RUN apt-get update -y && apt-get install -y fontconfig locales gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils libgbm-dev libgmp-dev libmpfr-dev libmpc-dev

# cache dotnet
RUN dotnet new console -o /tmp/dotnet && dotnet run --project /tmp/dotnet

# cache lean
RUN apt-get update && apt-get install -y zstd
RUN cd /root/sandbox/runtime/lean && \
    tar -I zstd -xf lean-4.10.0-rc2-linux_aarch64.tar.zst && \
    rm lean-4.10.0-rc2-linux_aarch64.tar.zst && \
    mv lean-4.10.0-rc2-linux_aarch64/* . && \
    rmdir lean-4.10.0-rc2-linux_aarch64

# python runtime
RUN . /home/ma-user/miniconda3/bin/activate && \
    cd /root/sandbox/runtime/python && \
    sed -i 's/\r$//' install-python-runtime.sh && \
    bash install-python-runtime.sh
# fix pyqt error, see https://github.com/NVlabs/instant-ngp/discussions/300
ENV QT_QPA_PLATFORM=offscreen
ENV OMP_NUM_THREADS=2

# final
WORKDIR /root/sandbox
ENV PATH="/home/ma-user/miniconda3/envs/sandbox-runtime/bin:${PATH}"
RUN poetry config virtualenvs.create false \
    && touch /home/ma-user/miniconda3/pyvenv.cfg \
    && poetry source add tuna https://pypi.tuna.tsinghua.edu.cn/simple --default \
    && poetry install \
    && cd ./docs \
    && npm ci \
    && npm run build

RUN useradd -m app && echo 'app:app' | chpasswd \
    && chmod og+rx /root \
    && mkdir -p /mnt \
    && chmod og+rwx /mnt

############################ server (addition) ############################

RUN mkdir -p /var/run/sshd
RUN apt-get update
RUN apt-get update --fix-missing && apt-get install -y wget g++ git make libdpkg-perl sudo vim unzip unrar openssh-server openssh-client ca-certificates psmisc screen --no-install-recommends
RUN rustup default stable
RUN apt-get install -y nginx
RUN apt-get install -y python3-pip
RUN python3 -m pip install supervisor
WORKDIR /root
