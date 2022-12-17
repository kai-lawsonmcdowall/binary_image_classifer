A (relatively) simple CNN to classify DNA double-stranded breaks as either "induced" or "not-induced". This leverages the power of the broad instituite's Integrated Genomics Viewer and batch scripting, which, when loaded with double strand break data, helps us to visualize the stark difference between high frequency endogenous breaks and the clean crispr (or other nuclease) induced cuts. I believe there is some potential here to leverage machine learning to help us make classifications on if break sites are induced or not, and help prevent the (current) manual necessity of identifying these by eye.

So far, I've built a very simple convolutional neural network with the following:

5 layers of:

convolutional 2D layer with 16 filters, a kernel of 3x3, the input size as our image dimensions, WxLx3, and the activation as ReLU. (have to check image dimensions) max pooling layer that halves the image dimension, so after this layer, the output will be 100x100x3.

loss function: binary crossentropy (generally used for binary classification tasks) optimization function: RMSProp (sensible alternative to adam as it auto-tunes the learning rate, however may switch over to ADAM as it may be perform better) train for 15 epochs

as of the 9th October, I'll only created the architecture, and haven't run it through it's paces, will do so once I label and preprocess some data.