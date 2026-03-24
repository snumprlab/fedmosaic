mkdir AQUA
wget https://researchdata.aston.ac.uk/380/1/SemArt.zip
unzip SemArt.zip
mv SemArt/Images AQUA
rm SemArt.zip
rm -rf SemArt
git clone https://github.com/noagarcia/ArtVQA
mv ArtVQA/AQUA/*.json AQUA
rm -rf ArtVQA
mv AQUA/Images AQUA/images

gdrive files download 1VLX4WwmE6tYhKK7vK1lSNZb5eFt64p53
tar -xvf AQUA_feddat_jsons.tar
mv train AQUA
mv test AQUA
rm AQUA_feddat_jsons.tar
