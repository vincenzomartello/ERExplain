# ExplainER

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

ExplainER is a package to explain any Entity Resolution model, using a Black-Box approach


# Key Features
The explanations can be provided into two different ways:
  - **Attribute-Level**: the system shows to the user a histogram reporting the importance of the attributes with respect to the prediction. The importance is a number in [0,1]
  - **Value-level**: the system shows to the user, for each chosen attribute set/single attribute A, the most influential values for A, with respect to the prediction
  - 
### Tech

Dillinger uses some well-know libraries:

* [Pandas] - To create perturbation
* [Numpy]
* [MLXtend] - To create explanations at value level

### Installation

To install the package simply run

```sh
$ pip install explainer
```



