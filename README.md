# VisualisingNNs
A little dive into visualising what is happening in neural nets based on distil.pub paper on feature visualisation.

This Code uses the MNIST dataset and trains a CNN to recognise the classes of digits. In addition it allows the user to visualise what the network is learning through multiple methods.

![FirstLayer](https://user-images.githubusercontent.com/37843140/171997293-259ebb46-553e-44e5-aa29-50f9d02c6553.png)

the image above shows the activations through simply plotting the weights in the first layers kernals (filters)
An important point to note about these images is that although the kernals are only 5x5 images you can see that they are detecting input in regions of basic dots and lines

![example input](https://user-images.githubusercontent.com/37843140/171997299-8440e31c-384b-4858-8b04-75a6751d2aee.png)

using an example image made in paint with size 28x28 as is used in MNIST, we can input this image, perform a forward pass to get the activations of said image and plot those activations at any layer of the network, not just the first.

![layer 1](https://user-images.githubusercontent.com/37843140/171997775-c359dd38-f6c0-463b-abc0-47513e4938da.png)
![layer 2](https://user-images.githubusercontent.com/37843140/171997810-28a4e28d-94fc-4560-af92-53f1c5badedc.png)
![MaxActivations](https://user-images.githubusercontent.com/37843140/171997295-399fddbd-9169-4648-b382-bc5b092548af.png)

finally as seen in the distil.pub paper https://distill.pub/2017/feature-visualization/ we can start from a random noisy image and use gradiant ascent to find the maximal activation image for each class at any layer of the network
although you cannot clearly see a stark number produced you can see the sensitive regions that input must be detected in to build the classifier. As seen in the earlier layers it builds from dots and lines to more complex shapes.
