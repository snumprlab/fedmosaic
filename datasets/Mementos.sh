git clone https://github.com/si0wang/Mementos.git
mkdir Mementos/train
mkdir Mementos/test
./gdrive files download 16OgGr7cb9egGzTGCao5xgWKmdaG06S6H
./gdrive files download 1rVEXDqUd-AX0uUSNd_Qq3MMxxPIDmncp
./gdrive files download 1Y3uH_Jw7zLRmvJIi1bE5DPcSBDA0xnDn
./gdrive files download 15KQZfWejTjyWyFYzSNyeB0jFtP0cSHh1
./gdrive files download 1Ze8H2B6am8xoZM6c7e7Z8j87fR0rYtum
./gdrive files download 1MXtRF_JG3DWKxH3WZf3OatlXIeQklrKo
unzip train_image_dailylife.zip
unzip train_image_robotics.zip
unzip train_image_comics.zip
mv image_robo Mementos/train
mv image_cmc Mementos/train
mv image_rw Mementos/train
unzip image_dailylife.zip
unzip image_robotics.zip
unzip image_comics.zip
mv image image_rw
mv image_robo Mementos/test
mv image_cmc Mementos/test
mv image_rw Mementos/test
rm train_image_dailylife.zip
rm train_image_robotics.zip
rm train_image_comics.zip
rm image_dailylife.zip
rm image_robotics.zip
rm image_comics.zip
