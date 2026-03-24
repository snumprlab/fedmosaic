./gdrive files download 193yftaEWSR8fwjvc9p-sqBsaZlVo4dNS
unzip Fashion200K.zip
rm Fashion200K.zip

./gdrive files download 1JFvucHGjKDBkjIhaBJ1hbrdWwZUDR6gQ
tar -xvf fashion200k_train_test.tar
mv fashion200k_train_test/* Fashion200K
rm -rf fashion200k_train_test
rm fashion200k_train_test.tar

./gdrive files download 1SoT4zScEf3g1Y0FeGpiL4YIk4A-ySLbN
tar -xvf FashionIQ.tar
rm FashionIQ.tar

./gdrive files download 15PD9O1wsLD-aA2N9lJVf4QNm2FjkmBmM
tar -xvf FashionIQ_train_test.tar
rm -rf FashionIQ/train
rm -rf FashionIQ/test
mv FashionIQ_train_test/* FashionIQ
rm FashionIQ_train_test.tar
rm -rf FashionIQ_train_test

