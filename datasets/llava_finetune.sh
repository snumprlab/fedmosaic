# 각 wget을 parellel하게 하는걸 추천함. 매우 오래걸림. 특히 vg랑 gqa

mkdir llava_finetune
cd llava_finetune

# original llava json file
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json?download=true
mv llava_v1_5_mix665k.json?download=true llava_v1_5_mix665k.json

# coco
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
cd..

# gqa
mkdir gqa
cd gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
rm images.zip
cd ..

# textvqa
mkdir textvqa
cd textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
rm train_val_images.zip
cd ..

# vg
mkdir vg
cd vg
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip images.zip
rm images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images2.zip
rm images2.zip
cd ..

# ocr_vqa
gdrive files download 17o7qLSvfLvjWZYH1C9rJ0m8TDms3qW9u
tar -xvf ocr_vqa.tar
rm ocr_vqa.tar

# splited json files
gdrive files download 1GHDpr9pvgrSUymwKFgVE8h0S0AsLOb87
tar -xvf llava_split_json.tar
rm llava_split_json.tar