# keras-example
Artificial Neural Net example in Python using Keras and TensorFlow. This neural net implements a classifier and operates on the 'moons' data set from the scikit-learn package.

To clone this repository:
```
git clone https://github.com/coledd/keras-example.git
```

After installing python3, you can install the following dependencies using pip. Alternatively, you may prefer to use your distribution's native package manager for the above packages.

```
python3 -m pip install --user tensorflow
python3 -m pip install --user keras
python3 -m pip install --user scikit-learn
python3 -m pip install --user matplotlib
python3 -m pip install --user numpy
python3 -m pip install --user scipy
```

You may want to do this in the keras-example directory if python2 is your default:
```
sed -e "1c\/usr/bin/python3" -i ./train.py
sed -e "1c\/usr/bin/python3" -i ./evaluate.py
```

To get a copy of the presentation that was given which included this example:
```
wget https://coledd.com/wp-content/uploads/2019/02/introduction_to_machine_learning.pdf
```

Here is the "moon data" that the model will train on:

![Moon Data](moon_data.png)

Here is a snapshot of the model loss over the 5000 epoch training:

![Training Loss](loss.png)

Here is the resultant decision boundary, with shading that represents the confidence near the boundary:

![Decision Boundary](decision_boundary.png)

