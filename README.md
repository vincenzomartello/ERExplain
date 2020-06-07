# ExplainER

ERExplain is a package to explain any Entity Resolution model, using a Black-Box approach


# Key Features
The explanations can be provided into two different ways:
  - **Attribute-Level**: the system shows to the user a histogram reporting the importance of the attributes with respect to the prediction. The importance is a number in [0,1]
  - **Value-level**: the system shows to the user, for each chosen attribute set/single attribute A, the most influential values for A, with respect to the prediction
  
## Usage

To use this package you need some simple requirements:
1. A model M per Entity Resolution (ER), for example Deepmatcher, but also simpler models, for example a decision tree
2. A Pandas dataframe D that contains labeled record pairs for ER and that as in this example:

|label      | ltable_product_title   | ltable_manufacturer | rtable_product_title | rtable_manufacturer|
| ----------- | ----------- |-----------|-----------|----------|
| 0      | Nvidia rtx 2060       | Asus        | Nvidia Rtx 2070 | MSI |
| 1   | Asus rog strix rtx 2080 super       | Asus | Asus Geoforce rtx 2060 s | Asus

The label 0 indicates a non-matching pair, while label 1 indicates matching pairs. Ltable_ and rtable_ are the prefix for the left and the right attributes respectively. You can use any prefix you prefer

3. A list of Pandas dataframes containing the sources of records, each of one read in the following way:

  ``` source1 = pandas.read_csv('source1_path',dtype=str) ```

3. A wrapper function that takes in input three parameters in this exact order: the dataframe D, the model M, and the list columns to ignore in D

### Tech

ERExplain uses some well-know libraries:

* [Pandas](https://pandas.pydata.org/): To create perturbation
* [Numpy](https://numpy.org/)
* [MLXtend](http://rasbt.github.io/mlxtend/): To create explanations at value level



### Installation

To install the package simply run

```sh
$ pip install erexplain
```



