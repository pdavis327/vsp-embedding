#!/bin/bash

# Step 1: Get existing machineset and create template yaml for new GPU machine

echo "Creating GPU machineset yaml resource"
MACHINE_SET_WORKER_NAME=$(oc get machinesets -n openshift-machine-api -o json | jq -r '.items[0].metadata.name')
oc get machineset $MACHINE_SET_WORKER_NAME -n openshift-machine-api -o yaml > ../rhoai-poc-chart/charts/rhoai-ai-poc-template/templates/infrastructure/machineset-gpu.yaml

# Get the name of the first MachineSet and store it in a variable to use as new GPU name.
GPU_NAME=$MACHINE_SET_WORKER_NAME-gpu

# Replace parameters w/ GPU params
sed -i '' \
    -e '/machine/ s/'"${MACHINE_SET_WORKER_NAME##*/}"'/'"${GPU_NAME}"'/g' \
    -e '/^  name:/ s/'"${MACHINE_SET_WORKER_NAME##*/}"'/'"${GPU_NAME}"'/g' \
    -e '/cluster-api-autoscaler/d' \
    -e '/uid:/d' \
    -e '/generation:/d' \
    -e '/resourceVersion:/d' \
    -e '/creationTimestamp:/d' \
    -e '/^status:/,/^[^ ]/{/^status:/d; /^[^ ]/!d; }' \
    -e 's|replicas: 0|replicas: {{.Values.GPU_REPLICAS}}|g' \
    -e 's/instanceType.*/instanceType: {{.Values.GPU_TYPE_AWS}}/' \
    "../rhoai-poc-chart/charts/rhoai-ai-poc-template/templates/infrastructure/machineset-gpu.yaml"
