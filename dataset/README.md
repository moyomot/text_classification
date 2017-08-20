# Download data
## dataset
https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

## pre-trained word embeddings
- GoogleNews-vectors-negative300.bin
```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

# Set data
```
dataset
|--ag_news_csv
|  |--classes.txt
|  |--readme.txt
|  |--test.csv
|  |--train.csv
|--embedding
|  |--GoogleNews-vectors-negative300.bin
```