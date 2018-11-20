$TF_CPP_MIN_LOG_LEVEL = 3
rm -r /tmp/test*

python3 classifier.py \
-d 288 4 \
-n true \
-p 0.95 \
-A None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test0

python3 classifier.py \
-d 288 288 4 \
-n true true \
-p 0 0 \
-A lrelu None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test1

python3 classifier.py \
-d 288 4 \
-n false \
-p 0.99 \
-A None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test2

python3 classifier.py \
-d 288 4 \
-n false \
-p 0.9 \
-A None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test3
