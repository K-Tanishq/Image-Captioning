# (Image Captioning for Visually Impaired Individuals)

# Problem Statement
Image captioning for the visually impaired is a significant issue that needs attention. It involves generating descriptions for images understandably and accurately. This task is crucial to aid visually impaired individuals in interpreting and understanding visual content. Despite advancements in technology, creating precise and contextually appropriate captions remains a challenging task. My project aims to address this problem and develop an efficient solution that can generate accurate and meaningful captions for images, thereby enhancing the experience for visually impaired individuals.
![Untitled (5)](https://github.com/K-Tanishq/Image-Captioning/assets/169484818/3c8312ce-8969-4805-addf-f303902a63c6)
![Untitled (4)](https://github.com/K-Tanishq/Image-Captioning/assets/169484818/972e204c-0fad-48c1-9810-27553ba343e1)
![Untitled (3)](https://github.com/K-Tanishq/Image-Captioning/assets/169484818/dba66397-4961-4f3e-9cca-3c269e618992)
![Untitled (2)](https://github.com/K-Tanishq/Image-Captioning/assets/169484818/8cf50c53-d633-4119-8fce-83ddc2da136c)

The main focus of my project is to generate an interpretable and meaningful set of captions for real-life images. I have also converted the generated captions to audio for the visually impaired to listen to.

# Motivation and Experiments Performed Before the Final Solution
### 1. General Method Towards the problem
The Encoder-Decoder architecture is the most general/common way I could think of to solve this problem. Here I use a CNN model to generate the embeddings from the image which are further used as the hidden state or as the first word token for the decoder RNN model.

![Untitled](https://github.com/K-Tanishq/Image-Captioning/assets/169484818/8fb66ff2-56a5-486e-9c21-b3fcc2b0ed5d)

The major problem with this approach is that as the length of the caption for an image becomes longer the decoder RNN models start to fail. It is known that RNNs are not good for capturing relations in sentences with long lengths due to the problem of vanishing gradient. Therefore, the last RNN blocks have the least info about the image as it is being used as a hidden state or the first word token at the first block of RNN.

### 2. Merge Architecture
Here I introduced a change in architecture. Instead of treating the architecture as an Encoder and Decoder, I thought of doing the work of CNN and RNN in parallel and then adding the image and text embeddings that I get from CNN and RNN respectively then passing it through the Linear layer to generate the caption.

![Untitled (1)](https://github.com/K-Tanishq/Image-Captioning/assets/169484818/66750e85-221a-4226-8cf1-d1847009e59a)

The method is trained on the next word prediction task to generate the captions i.e., given n-1 words for a caption we try to predict the nth word.

This method solves the problem of image importance not being propagated through time to the last RNN block. Now I have an embedding that is richer than the previous embedding since it captures the details of the image with every part of the newly formed embedding.

### 3. Use of VAE
Now to improve the above-defined architecture I tried to use a better image representation (image embedding) for which I took the help of VAE to generate better image representation.

So, to get the image embeddings or (latent embeddings) for image captioning I trained the VAE as my encoder-CNN and its reparameterization trick allows me to get better embeddings because of the continuous nature of its latent space. I used these embeddings and added them to the text embeddings to get total embeddings from which I generated my captions of the image.

![Untitled (6)](https://github.com/K-Tanishq/Image-Captioning/assets/169484818/db28db27-dc74-4b54-9f89-cf86ba311db1)

### 4. Use of GAN’s Intuition
As soon as I think of a generation task, the first thing that pops up in my mind is GANs or (Generative Adversarial Networks). Hence I tried to make a generator model that would help me do the same task by just taking the image and generating labels. But, I didn’t quite succeed here since the discriminator overpowered every generator model that I could think of.

### 5. Motivation
The previously presented solutions provide insight into generating meaningful captions. In the approaches defined above I am trying to generate captions based on how good the image embedding or the text embedding were. Let’s take a moment and try to think about what defines a good caption. A good caption is one whose text embedding is as close to the image embedding as possible, i.e., the cosine similarity between the embeddings is close to one.

This led me to think about why I did not try to maximize this cosine similarity between the text embedding and the image embedding. This takes me to my solution as stated in the following section.

The above figure describes what I was trying to convey through my motivation. Let‘s say we have the image embedding (Image-1) and text embedding (Embed-1) as shown in the above figure and respectively for other data points as well. Now to predict good captions the aim of my model should be to get values of diagonal elements as close to one as possible and other elements of the diagonal matrix to be as close to zero as possible.

This gave me an idea to implement a model which gives the similarity matrix as defined above. To calculate the loss I have used the cross-entropy loss of all rows and columns of the similarity matrix with a zeros tensor containing one only at those specific row or column index respectively. Then I have backpropagated the loss to train the model.
