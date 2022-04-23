pip install -qq git+https://github.com/rwightman/pytorch-image-models
pip install -qq albumentations==1.1.0
pip install -qq ttach
python inference.py --file /Cultivar_FGVC9/cfg/tf_efficientnet_b4_ns_inference.yaml
