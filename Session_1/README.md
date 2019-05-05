
## Assignment 1A
***

> <a href="https://colab.research.google.com/github/hardayal/EVA/blob/master/Session_1/EVA_Assignment_1A.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

> GitHub: <a href="https://github.com/hardayal/EVA/blob/master/Session_1/EVA_Assignment_1A.ipynb">EVA_Assignment_1A.ipynb</a>

***
## Assignment 1B
***
### What are Channels and Kernels (according to EVA)?
***

>####
> Kernel: A kernel is used to extract a particular feature from an object. Eg: vertical lines or horizontal lines or curved lines will all be 3 different kernels. Another example could be to extracting rice grains(from a prepared dish) using a rice grain kernel or extracting peas(from a prepared dish) using a kernel.

> Channel: A channel however is like a bucket storing extracted features of a particular type. example: all the vertical lines or all the horizontal lines extracted from a particular kernel will create a channel.

***
### Why should we only (well mostly) use 3x3 Kernels?
***

>####
> We have a choice to choose kernels of different sizes. A 3X3 kernel is largely used because:
> A) The features that would be extracted will be more local (and less generic) in a smaller kernel than in a larger kernel.
> B) The amount of information or features extracted will be more, which can be further useful in later layers.3) NVIDIA provides better, faster and more optimized computations for a 3X3 kernel than a kernel of larger size.

***
### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
***
> 199->(3x3)->197->(3x3)->195->(3x3)->193->(3x3)->191->(3x3)->189->(3x3)->187->(3x3)->185->(3x3)->183->(3x3)->181->(3x3)->179->(3x3)->177->(3x3)->175->(3x3)->173->(3x3)->171->(3x3)->169->(3x3)->167->(3x3)->165->(3x3)->163->(3x3)->161->(3x3)->159->(3x3)->157->(3x3)->155->(3x3)->153->(3x3)->151->(3x3)->149->(3x3)->147->(3x3)->145->(3x3)->143->(3x3)->141->(3x3)->139->(3x3)->137->(3x3)->135->(3x3)->133->(3x3)->131->(3x3)->129->(3x3)->127->(3x3)->125->(3x3)->123->(3x3)->121->(3x3)->119->(3x3)->117->(3x3)->115->(3x3)->113->(3x3)->111->(3x3)->109->(3x3)->107->(3x3)->105->(3x3)->103->(3x3)->101->(3x3)->99->(3x3)->97->(3x3)->95->(3x3)->93->(3x3)->91->(3x3)->89->(3x3)->87->(3x3)->85->(3x3)->83->(3x3)->81->(3x3)->79->(3x3)->77->(3x3)->75->(3x3)->73->(3x3)->71->(3x3)->69->(3x3)->67->(3x3)->65->(3x3)->63->(3x3)->61->(3x3)->59->(3x3)->57->(3x3)->55->(3x3)->53->(3x3)->51->(3x3)->49->(3x3)->47->(3x3)->45->(3x3)->43->(3x3)->41->(3x3)->39->(3x3)->37->(3x3)->35->(3x3)->33->(3x3)->31->(3x3)->29->(3x3)->27->(3x3)->25->(3x3)->23->(3x3)->21->(3x3)->19->(3x3)->17->(3x3)->15->(3x3)->13->(3x3)->11->(3x3)->9->(3x3)->7->(3x3)->5->(3x3)->3->(3x3)->1

So we need to perform 100 3X3 convolution operations to reach 1X1 from 199X199

***
