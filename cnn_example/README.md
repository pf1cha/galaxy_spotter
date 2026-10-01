# Galaxy Spotter

Training time on CPU for me: around 30 minutes for 5 epoch

## File structure

- config.py - information about your parameters for the model and path to data
- dataset.py - here you can find dataset class, where we load image, preprocess it and return pytorch tensor with label of the class
- evaluate.py - in that class we evaluate our model on unseen dataset during training process. 
- galaxy_cnn.py - out CNN model architecture
- save_model.py - saving parameters for our application
- train.py - training model
- visualization.py - plot a few examples after training the model

## Different metrics for model

### Elliptical Accuracy

Elliptical Accuracy measures the proportion of true elliptical galaxies that are correctly classified as elliptical.

$$
\text{Elliptical Accuracy}
=
\frac{\text{Correctly Predicted Elliptical}}
{\text{Total True Elliptical}}
$$

### Spiral Accuracy

Spiral Accuracy measures the proportion of true spiral galaxies that are correctly classified as spiral.

$$
\text{Spiral Accuracy}
=
\frac{\text{Correctly Predicted Spirals}}
{\text{Total True Spirals}}
$$

### Balanced Accuracy

Balanced Accuracy is the arithmetic mean of the two class-specific accuracies:

$$
\text{Balanced Accuracy}
=
\frac{
\text{Elliptical Accuracy}
+
\text{Spiral Accuracy}
}{2}
$$
