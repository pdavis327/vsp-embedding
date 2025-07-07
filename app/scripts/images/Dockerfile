FROM quay.io/opendatahub/workbench-images@sha256:8b24def43c789db6d0dd5fea4fc61b5959661d1d5e9731d57be107bad07477a3

COPY requirements.txt /opt/app-root/bin/

RUN pip install -r /opt/app-root/bin/requirements.txt && rm /opt/app-root/bin/requirements.txt
