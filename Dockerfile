# Use Python >= 3.6
FROM python:3.8

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY ./ /

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1 -y

RUN apt-get update && apt-get install python3-dev graphviz libgraphviz-dev pkg-config -y

RUN pip3.8 install --upgrade pip
RUN pip3.8 install wheel
RUN pip3.8 install --no-cache-dir -r requirements.txt
RUN python3.8 /driver/model_build.py --yaml_path /driver/config.yaml --raw_dataset_path /dataset/TJ-RTL-toy --data_pkl_path /driver/dfg_tj_rtl.pkl --graph_type DFG

CMD python3.8 /driver/detect.py --yaml_path /driver/config.yaml --raw_dataset_path /dataset/TJ-RTL-toy --data_pkl_path /driver/dfg_tj_rtl.pkl --graph_type DFG