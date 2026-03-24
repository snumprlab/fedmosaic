# Knowledge Grounded QA
# multimodalQA
# ./gdrive files download 1tzjY2FQUGWzFLkONrPlGZ-TL_DY8JFnO
# # manymodalQA
# ./gdrive files download 1NM5ieGcE-AX2LIrHgTw9QB3k9d-0vYel
# mkdir KGQA

# unzip ManyModalQA.zip
# unzip MultiModalQA.zip

# mv ManyModalQA/ KGQA/
# mv MultiModalQA/ KGQA/


# rm ManyModalQA.zip
# rm MultiModalQA

# WebQA
gdrive files download 1uskKtaMtKVqjMX9cjPZ70c3I9q2Xux4e
unzip WebQA.zip
rm WebQA.zip

gdrive files download 1yrKz3_aF320FUY0Pzuc43YyOliYV8A8L
unzip TQA.zip
rm TQA.zip

mkdir AQUA
wget https://researchdata.aston.ac.uk/380/1/SemArt.zip
unzip SemArt.zip
mv SemArt/Images AQUA
rm SemArt.zip
rm -rf SemArt
git clone https://github.com/noagarcia/ArtVQA
mv ArtVQA/AQUA/*.json AQUA
rm -rf ArtVQA
cd AQUA
mv Images images

gdrive files download 1jXip1cW30g7g7MfXvb4cI2iGa_qqgYDM
tar -xvf dvqa.tar
rm dvqa.tar
