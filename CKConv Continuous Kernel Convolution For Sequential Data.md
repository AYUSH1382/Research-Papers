# CKConv: Continuous Kernel Convolution For Sequential Data


### Introduction

Recurrent Neural Networks (RNNs) have long governed tasks handling sequential data. In practice, however, the effective memory horizon of RNNs, i.e., the number of steps the network can retain information from, has proven to be surprisingly small due to the vanishing gradients problem. 

Convolutional networks (CNNs) have proven a strong alternative to recurrent architectures for several tasks. CNNs avoid the training instability and vanishing / exploding gradients characteristic of RNNs by circumventing Back-Propagation Through Time (BPTT). These architectures parameterize their convolutional kernels as a sequence of independent weights so their memory horizon must be defined a priori.

The Continuous Kernel Convolutions (CKConvs) has the following properties:

a) CKConvs are able to consider arbitrarily large memory horizons within a single operation.

b) CKConvs do not rely on any form of recurrency.

c) CKCNNs detach the memory horizon –often referred to as receptive field– from (i) the depth of the network, (ii) the dilation factor used, and (iii) the parameter count of the architecture.

d) CKCNNs do not make use of Back-Propagation Through Time.

e) CKConvs easily handle irregularly sampled data as well as data sampled at different sampling rates since they can be evaluated at arbitrary positions.

### Related Work

Vanishing and exploding gradients are a long-standing problem for recurrent networks and they have tried working on it to lessen it. There are three class of problems that aims to alleviate the problem. The first class is gating mechanisms, second class is unitary recurrent units, third class is convolutional networks (CNNs). CNNs elude exploding and vanishing gradients by avoiding recurrent connections and Back-Propagation Through Time altogether. CKConvs are able to handle arbitrarily large sequences under a fixed parameter budget.

Implicit neural representations aim to represent data by encoding it in the weights of a neural network. The fact is that these convolutional kernels are not known a priori but learned as part of the optimization task of the CNN. Continuous formulations to convolutional kernels were introduced as a powerful alternative to handle irregularly sampled 3D data.

### The Convolution Operation

They have used a few representations $[n]$ the set ${0, 1, 2, . . . , n}$. Bold capital and lowercase letters depict vectors and matrices, sub-indices are used to index vectors,and parentheses are used for time indexing. In Centered and Causal Convolutions the convolution is defined as:
$$
(x ∗ ψ)(t) =
N_C
∑
c=1
∫_
R
x_c(τ )ψ_c(t − τ ) dτ
$$
The convolution is effectively performed between the input signal described as a sequence of finite length and a convolutional kernel. The convolutional kernel is commonly centered around the point of calculation t which is solved by providing a causal formulation to the convolution.

In Convolutions with Discrete Kernels the memory horizon NK must be defined a priori. In order to alleviate the limitations mentioned previously, it has been proposed to dilate the sequence parameterizing the convolutional kernel K by a factor η. Dilated convolutions are unable to model functions depending on input values within $x(ητ)$ and  $x(η(τ + 1))$. By carefully selecting the dilation factor at every layer, one can guarantee that some kernel hits each input within the memory horizon of the network.

### Continuous Kernel Convolution

They have tried formulating the convolutional kernel ψ as a continuous function parameterized by a small neural network MLP^ψ^. They have been able to construct global memory horizons without modifying the structure of the network or adding more parameters. CKConvs are able to handle irregularly sampled and partially observed data natively. To this end, it is sufficient to sample MLP^ψ^ at positions for which the input signal is known and perform the convolution operation with the sampled kernel. CKCNNs can be deployed at sampling rates different than those seen during training, and it can be trained on data with varying temporal resolutions. It is seen that linear recurrent units can be described as a CKConv with a particular family of convolutional kernels: exponential functions.  Since exponentially growing gradients lead to divergence, the eigenvalues for converging architectures are often smaller than 1. This explains why the effective memory horizon of recurrent networks is so small.

Linear recurrent units are a convolution between the input and a very specific class of convolutional kernels: exponential functions. It is observed discrete convolutions use a predefined, small kernel size, and thus possess a restricted memory horizon. CKConvs, on the other hand, are able to define arbitrary large memory horizons. It seen that CKConvs are a generalization of (linear) recurrent architectures which allows for parallel training and enhanced expressivity.

The convolutional kernel MLP^ψ^ is parameterized by a conventional L-layer neural network. Having a closer look at their approach it can be thought of as providing implicit neural representations to the unknown convolutional kernels ψ of a conventional convolutional architecture. Conventional applications assume the distribution of the input features to be centered around the origin. This is orthogonal to implicit neural representations, where the spatial distribution of the output, i.e. , the value of the function being implicitly represented, is uniformly distributed. Consequently,it is seen conventional initialization techniques lead to poor performance.

The expressiveness of such an approximation is determined by the number of knots the basis provides, i.e., places where a non-linearity bends the space. Naturally, the better the placing of these knots at initialization, the faster the approximation may converge. For a spatially uniform distributed input, the knots should be uniformly distributed as well.

It is observed that finding an initialization with an exponential number of knots is a cumbersome and unstable procedure so to cater to that issue they have utilized an initialization procedure with which the total number of knots is equal to the number of neurons of the network. It is observed ReLU networks show large difficulties in representing very nonlinear and non-smooth functions.

Sine layers are much more robust to parameter selection, and can be tuned to benefit pattern approximation at arbitrary –or even multiple– positions in space. Fourier transform states that any integrable function can be described as a linear combination of an infinite basis of phase-shifted sinusoidal functions. Intuitively, approximations via Sine networks can be seen in terms of an exponentially large Fourier-like basis. The exponential growth combined with the periodicity of sine allows for astonishingly good approximations: the more terms in a Fourier transform, the better the approximation becomes.

As the target function is defined uniformly on a given interval, uniformly initializing the knots of the spline basis provides faster and better approximations. If all the knots are initialized at zero, the best approximation at initialization is given by a straight line.

### Experiments

They have validated their approach across a large variety of tasks and against a large variety of existing models. They have parameterized all convolutional kernels as a 3-layered MLP with Sine non linearities. They have used the convolution theorem to speed up convolution operations in the networks with the Fourier transform.  Stress experiments - It is validated that the memory horizon of shallow CKCNNs is not restricted by architectural choices. It is seen that a shallow CKCNN solves both problems for all sequence lengths considered without structural modifications.

Discrete sequences - They validated the applicability of CKConvs for discrete sequence modeling tasks: sMNIST, pMNIST and sCIFAR10. It is observed that shallow CKCNNs outperform strong recurrent and convolutional models. A small CKCNN obtains state-of-the-art on sMNIST and wider CKCNN also increases the results on the datasets.

Time-series modeling - They have tried evaluating CKCNN on time series data and on long-term dependencies. It is seen that it performs well and has the ability to handle very long term dependencies. Testing at different sampling rates - In this part they have considered data that is trained at different sampling rates. It is seen that the performance of CKCNNs remains relatively stable even for large sampling rate fluctuations and it even outperforms HiPPO. Irregularly-sampled data - At last they even explored the applicability of CKCNNs  for irregularly-sampled data. It exhibits stable performance but does cross NCDEs.

### Conclusion

Further experiments indicate that the models do not benefit from larger depth, which suggests that CKCNNs do not rely on very deep features. CKCNNs can be executed in parallel, and thus can be much faster than recurrent networks. MLPs parameterizing spatial functions should use sine nonlinearities. The kernels often contain frequency components higher than the resolution of the grid used during training.
