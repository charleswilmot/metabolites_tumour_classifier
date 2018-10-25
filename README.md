# Implementation of a multilayer perceptron for classifying spectrum.
## Requirements

This framework requires tensorflow

You can check if tensorflow is installed by running

`python3 -c "import tensorflow"`

You can install it with

`sudo pip3 install tensorflow` if you have sudo access or
`sudo pip3 install --user tensorflow` otherwise

To generate the documentation, you will also need doxygen. This is installed on most systems though.

## Documentation

Please have a look at the documentation.

You can generate it and open it with

```
cd doc
doxygen ./doxygen_conf_file.conf
firefox ./html/index.html
```

Please also have a look at the help option of the command line:

```
cd src
python3 classifier.py -h
python3 classifier.py train -h
python3 classifier.py test -h
```
