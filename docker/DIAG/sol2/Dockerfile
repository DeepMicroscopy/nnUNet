FROM doduo1.umcn.nl/uokbaseimage/diag:latest

# Additional packages
RUN apt-get update && apt-get install -y libgeos-dev
RUN python3 -m pip install shapely==1.7.1
RUN python3 -m pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main
RUN python3 -m pip install albumentations==1.2.1

# nnUNet_v1
RUN python3 -m pip install git+https://github.com/DIAGNijmegen/nnUNet@feature/pathology-augmentations 

# nnUNet_v2
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/DIAGNijmegen/nnUNet_v2 /home/user/nnunet && \
    git -C /home/user/nnunet checkout rework && \
    chown -R user /home/user/nnunet && \
    pip3 install -e /home/user/nnunet

USER root

# Make sure you install the latest version (some uses of a dockerimage may override the entrypoint) 
ENTRYPOINT ["pip3 install --no-use-pep517 -e /data/pathology/projects/nnUNet_v2/nnUNet_v2/"] 

# # Update pixman 
# RUN : \ 
#     && sudo apt-get update \
#     && sudo apt-get install meson \ 
#     && wget https://www.cairographics.org/releases/pixman-0.42.2.tar.gz \ 
#     && tar -xf pixman-0.42.2.tar.gz \ 
#     && cd pixman-0.42.2 \ 
#     && mkdir build \ 
#     && cd build \ 
#     && meson --prefix=/usr --buildtype=release \ 
#     && ninja \ 
#     && sudo ninja install

