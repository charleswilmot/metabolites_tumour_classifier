export TF_CPP_MIN_LOG_LEVEL=3
rm -r /tmp/test*

python3 classifier.py \
-d 784 200 10 \
-n false false \
-p 0 0.5 \
-A lrelu None \
-b 450 \
-vvvv train -L mse -S 10 -e 300 -t 1000 ../tmp/mnist/ /tmp/test0

#python3 classifier.py \
#-d 784 784 10 \
#-n true true \
#-p 0 0 \
#-A lrelu None \
#-b 450 \
#-vvvv train -L mse -S 10 -e 30 -t 100 ../tmp/mnist/ /tmp/test1

python3 classifier.py \
-d 784 10 \
-n false \
-p 0.1 \
-A None \
-b 450 \
-vvvv train -L mse -S 10 -e 30 -t 100 ../tmp/mnist/ /tmp/test2

python3 classifier.py \
-d 784 10 \
-n false \
-p 0.2 \
-A None \
-b 450 \
-vvvv train -L mse -S 10 -e 30 -t 100 ../tmp/mnist/ /tmp/test3
