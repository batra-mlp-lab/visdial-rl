#!/usr/bin/env bash

# Processed dialog data for VisDial v0.5
wget -P data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_data.h5
wget -P data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_data_gencaps.h5
wget -P data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_params.json

# Processed image features for VisDial v0.5, using VGG-19
wget -P data/visdial/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/data_img.h5
