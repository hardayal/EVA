## Assignment 4: Architectural Basics


>#### 


#### 1. How many layers
***

>Based on CPU/GPU requirements we need to decide how many parameter we can use. 
Based on number of parameters and the accuracy (usecase) we can decide the number of layers.
#### 2. 3x3 Convolutions
>Best and Optimal Kernal, feature extractor.
#### 3. Kernels and how do we decide the number of kernels?
>Based on the number of classification or amount of information needed and hardware available.
#### 4. Receptive Field
>This will give total number of pixels that our kernals have seen. 
ex: If we use two 3*3 kernels then Receptive fields = 5*5 
#### 5. MaxPooling
>We can think of MaxPooling when we want to reduce the number of channels, this inturn reduces the parameters as well
#### 6. Position of MaxPooling
>Minimum 2 layers away from the final/decisition makeing layer, and also maxpooling should be applied atleast after 3 layers of convolution.
#### 7. The distance of MaxPooling from Prediction
>Minimum 2 layers
#### 8. 1x1 Convolutions
>Merging the similar features and create the complex channels, we use this to reduce the number of channels currently, it needs less computation an less parameters
#### 9. Concept of Transition Layers
>Transition layers are such layers in convolution block where depth of feature maps remain constant but number of channel could be changed in this block.

#### 10. Position of Transition Layer
>After we reach receptive fields of atleast 7*7 in small images ex: 28*28 and atleast 11*11 in case of big images ex: 400*400
#### 11. SoftMax
>Activation function which will exaggerate the output.
#### 12. Batch Size, and effects of batch size
>On increase in batch size accuracy will improves as backpropagetion will have average value of all the classes
#### 13. Batch Normalization
>it puts the data set in range of 0 to 1 and deal with the problem of gradient explosion.
#### 14. The distance of Batch Normalization from Prediction
>Batch normalization should be used before prediction
#### 15. Image Normalization
>It is used to increase contrast of the pixels for effectively extraction of features. 
#### 16. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
>When our image is 7*7 or less as our kernel will cover some pixels more time than others. 
#### 17. How do we know our network is not going well, comparatively, very early
>Comparing the output of first four Epochs
#### 18. DropOut
>To reduce the over fitting, and to balckout some kernels so that other kernal learning improves.
#### 19. When do we introduce DropOut, or when do we know we have some overfitting
>To bring down the difference between training accuracy and validation accuracy.
High training accuracy and low validation accuracy = overfitting
#### 20.Learning Rate
>This has to be optimal, as of now we reduce the learning rate gradually
#### 21. LR schedule and concept behind it
>We can use this shedule to reduce the LR only when test accuracy is constant or not improvig.
#### 22. Number of Epochs and when to increase them
>If the validation accuracy is improving then we can increase numer of Epochs to see where it reached highest accuracy value.
#### 23. When to add validation checks
>When we want to see which epoch will give better result.
#### 24. Adam vs SGD
>Optimization functions