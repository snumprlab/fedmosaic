gdrive files download 1g5iuAvYGEExk55Z8vfcQ7LoookkIkFOb
tar -xvf Bongard-HOI.tar
rm Bongard-HOI.tar

gdrive files download 1aXr3ihVq0mtzbl6ZNJMogYEyEY-WALNr
unzip images.zip 
git clone https://huggingface.co/datasets/joyjay/Bongard-OpenWorld
mv images Bongard-OpenWorld
rm images.zip
cd Bongard-OpenWorld 
rm -rf .git
rm README.md

