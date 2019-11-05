A interface to convert youtube videos into chatbots

# To create pix2pix model

# export the model in pix2pix-tensorflow root folder
python pix2pix.py --mode export --output_dir export/ --checkpoint ckpt/ --which_direction BtoA

# port the model to tensorflow.js
cd server
cd static
mkdir models
cd ..
python3 tools/export-checkpoint.py --checkpoint ../export --output_file static/models/MY_MODEL_BtoA.pict