## @package configreader
#  Tools to load a model from a config file\n
#  It follows the pep8 style convention defined [here](https://www.python.org/dev/peps/pep-0008/)\n
import os
import numpy as np
import configparser
import tensorflow as tf


def lrelu(x):
    a = 1 / 3
    return tf.nn.relu(x) * (1 - a) + a * x


## Tool class to read parameters from a config file
class Configurator:

    ## The constructor
    #  @param filepath path to an existing config file
    def __init__(self, filepath):
        self.filepath = filepath
        self.conf = configparser.SafeConfigParser()
        self.conf.read(self.filepath)
        self.conf_dict = self.to_dict(self.conf)
        self.check()

    ## Parses config object for parameter key and value extraction to get
    #  a dictionary holding all vales in proper types.
    #  @param parser <VAR>configparser</VAR> object to use for extraction
    #  @returns cofiguration dictionary
    def to_dict(self, parser):
        d = {}
        for section, content in parser._sections.items():
            d[section] = {}
            for key, value in content.items():
                try:
                    d[section][key] = eval(value)
                except (NameError, SyntaxError) as e:
                    d[section][key] = value
        return d

    ## Getter that returns config values
    #  @param key name of requested <VAR>key's</VAR> value
    #  @returns respective <VAR>value</VAR> of given <VAR>key</VAR>, section dictionaries
    #           in general.
    def __getitem__(self, key):
        return self.conf_dict[key]

    ## Writes a configurator object to a file object
    #  @param file_object file_object to write file into
    def write(self, file_object):
        self.conf.write(file_object)

    ## Helper function for printing messages
    # it transforms "string_1", "string_2" into "[string_1][string_2]:"
    #  @param cat string number one
    #  @param subcat string number two
    def cat_2_string(self, cat, subcat):
        return "[\"" + cat + "\"][\"" + subcat + "\"]:"

    ## Helper function that returns an Exception with an appropriate message
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    #  @param message error message that the exception should raise
    def cat_2_simple_exception(self, cat, subcat, message):
        return Exception(self.cat_2_string(cat, subcat) + " {} {}".format(self[cat][subcat], message))

    ## Helper function that check wether the field corresponding to cat/subcat
    # is a directory
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    def check_is_dir(self, cat, subcat):
        if not os.path.isdir(self[cat][subcat]):
            raise self.cat_2_simple_exception(cat, subcat, "not a directory.")

    ## Helper function that check wether the field corresponding to cat/subcat
    # is a directory containing files
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    def check_not_empty(self, cat, subcat):
        if not os.listdir(self[cat][subcat]):
            raise self.cat_2_simple_exception(cat, subcat, "is empty.")

    ## Helper function that check wether the field corresponding to cat/subcat
    # points to a path that exists
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    def check_path_exists(self, cat, subcat):
        if not os.path.exists(self[cat][subcat]):
            raise self.cat_2_simple_exception(cat, subcat, "does not exist.")

    ## Helper function that check wether the field corresponding to cat/subcat
    # is the path to a file, and not a directory
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    def check_isfile(self, cat, subcat):
        if not os.path.isfile(self[cat][subcat]):
            raise self.cat_2_simple_exception(cat, subcat, "is not a file.")

    ## Helper function that check wether the field corresponding to cat/subcat
    # is a valid class name
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    #  @param from_name a dictionary that maps a class names as strings to the acctual class
    def check_class_exists(self, cat, subcat, from_name):
        if not self[cat][subcat] in from_name:
            raise self.cat_2_simple_exception(cat, subcat, "is not a valid class name.")

    ## Helper function that check wether the field corresponding to cat/subcat
    # exists in the onfig file
    # this implementation is @deprecated, use @see verify_key_exists instead
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    def check_key_exists(self, cat, subcat=None):
        if subcat is None:
            try:
                ret = cat in self
                return ret
            except Exception as e:
                return False
        else:
            try:
                ret = subcat in self[cat]
                return ret
            except Exception as e:
                return False

    ## Helper function that check wether the field corresponding to cat/subcat
    # exists in the onfig file
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    def verify_key_exists(self, cat, subcat=None):
        message = " is obligatory parameter but not defined in config."
        if not cat in self.conf_dict:
            raise self.cat_2_simple_exception(cat, subcat, message)
        if subcat is not None and not subcat in self.conf_dict[cat]:
            raise self.cat_2_simple_exception(cat, subcat, message)

    ## Helper function that check wether the field corresponding to cat/subcat
    #  is an instance of one of the types / classes passed in instances
    #  @todo clean the code of that function, remove the "required" param, there is obvioulsy a design issue...
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    #  @param instances a list of types / classes
    def check_is_instance(self, cat, subcat, instances, required=True):
        if required:
            self.verify_key_exists(cat, subcat)
        else:
            if not self.check_key_exists(cat, subcat):
                pass
        if not isinstance(self[cat][subcat], instances):
            name_list = ", ".join([i.__name__ for i in instances]) if isinstance(instances, tuple) else instances.__name__
            must_be = "must be a " if len(name_list) == 1 else "must be one of "
            raise self.cat_2_simple_exception(cat, subcat, must_be + name_list)

    ## Helper function that check wether the field corresponding to cat/subcat
    #  is an iterator that contains only object which type is in instances
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    def check_is_instance_all(self, cat, subcat, instances):
        for item in self[cat][subcat]:
            if not isinstance(item, instances):
                name_list = ", ".join([i.__name__ for i in instances])
                if len(name_list) == 1:
                    must_be = "at least one element is not a "
                else:
                    must_be = "at least one element is not one of "
                raise self.cat_2_simple_exception(cat, subcat, must_be + name_list)

    ## Helper function that check wether the field corresponding to cat/subcat
    #  satisfy a condition
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    #  @param func the condition to be satisfied as a function that returns a bool
    #  @param message the error message
    def check_value(self, cat, subcat, func, message):
        if not func(self[cat][subcat]):
            raise self.cat_2_simple_exception(cat, subcat, message)

    ## Helper function that check wether the field corresponding to cat/subcat
    #  is an iterator that contains only objects that satisfy a condition
    #  @param cat id of one category (section of the config file), string
    #  @param subcat id of one sub-category (sub-section of the config file), string
    #  @param func the condition to be satisfied as a function that returns a bool
    #  @param message the error message
    def check_value_all(self, cat, subcat, func, message):
        for item in self[cat][subcat]:
            if not func(item):
                raise self.cat_2_simple_exception(cat, subcat, message)

    ## Helper function that check wether the field corresponding to cat/subcat
    # is less than an other one
    #  @param cat1 id of one category (section of the config file), string
    #  @param subcat1 id of one sub-category (sub-section of the config file), string
    #  @param cat2 id of one category (section of the config file), string
    #  @param subcat2 id of one sub-category (sub-section of the config file), string
    def check_inferior(self, cat1, subcat1, cat2, subcat2):
        if not self[cat1][subcat1] <= self[cat2][subcat2]:
            raise Exception(self.cat_2_string(cat1, subcat1) + "must be <=" + self.cat_2_string(cat2, subcat2))

    def check(self):
        cat = "network"
        self.check_key_exists(cat)
        self.check_is_instance(cat, "layers_sizes", list)
        self.check_is_instance_all(cat, "layers_sizes", int)
        self.check_value_all(cat, "layers_sizes", lambda x: x > 0, "layers sizes must be greater than 0.")
        self.check_value(cat, "layers_sizes", lambda x: len(x) > 1, "must contain at least 2 entries")
        n_layers = len(self[cat]["layers_sizes"]) - 1
        self.check_is_instance(cat, "batch_norm", list)
        self.check_is_instance_all(cat, "batch_norm", bool)
        self.check_value(cat, "batch_norm", lambda x: len(x) == n_layers, "length must correspond with the number of layers")

        self.check_is_instance(cat, "dropout", list)
        self.check_is_instance_all(cat, "dropout", (float, int))
        self.check_value(cat, "dropout", lambda x: len(x) == n_layers, "length must correspond with the number of layers")
        self.check_value_all(cat, "dropout", lambda x: 0 <= x < 1, "dropout value must be in [0, 1[")

        self.check_is_instance(cat, "activation_functions", list)
        self.check_value(cat, "activation_functions", lambda x: len(x) == n_layers, "length must correspond with the number of layers")
