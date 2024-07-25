FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN useradd -ms /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R vscode:vscode /workspace
USER vscode

CMD ["/bin/bash"]
