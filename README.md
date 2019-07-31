# Imhotep
Imhotep was the first physician in history, and the goal of this project
is to create the first "physician" that patients will consult, namely an
AI app that will tell the patient whether or not their symptoms warrant an
emergency room visit. To achieve this, a Multi-Layer Perceptron trained on
a dataset taken from the CDC's NAMHCS survey. After training, the network
was fed several sets of vital signs and symptoms, and ultimately, the AI
consistently predicted about 66% of cases correctly. While better than
chance, this was not sufficient to continue the project, and this project is
no longer being maintained. Feel free to fork it if you have any ideas for
improving the AI, but preliminary data visualization using PyPlot suggests
that the data simply do not allow for any clear delineations of "truly sick"
vital signs that are distinguishable from urgent care/non-urgent situations;
that is to say, the distributions appear to overlap.

## Getting Started
To run this project, you need PyTorch, NumPy, and Pandas. With those
installed, clone this repository, open a command-line window in the cloned
directory, and run `python smalltorch.py`. It's called `smalltorch.py`
because I chose to undersample from a skewed training dataset to get a 
50:50 split in my training data.

## Future Directions
The AI is currently trained on approximately 10,000 sets of training data,
from 3 years of combined data from the NAMHCS. You may consider adding more
data, or altering the structure of the neural network, or some advanced
techniques that I am unaware of. As far as I can tell, the most promising
direction to take this would be to try to reduce the false negative rate as
much as possible, in spite of some increase in the false positive rate. I
believe you can add a `weights` argument to the `SoftmaxCrossEntropy` to
achieve this
