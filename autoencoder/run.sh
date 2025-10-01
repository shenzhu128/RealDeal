# To train the autoencoder, run the following command:
# Adjust the --data_path argument as needed.
python AutoencoderHCP3D.py \
--data_path "/path/to/your/data"

# To encode all inputs using the provided checkpoint, run:
# Adjust the --data_path and --checkpoint arguments as needed.
python EncodeHCP.py \
--data_path "/path/to/your/data" \
--checkpoint "/path/to/autoencoder_checkpoint.pth"

# To decode all encoded inputs using the provided checkpoint, run:
# Adjust the --data_path and --checkpoint arguments as needed.
python DecodeHCP.py \
--data_path "/path/to/your/data" \
--checkpoint "/path/to/autoencoder_checkpoint.pth"