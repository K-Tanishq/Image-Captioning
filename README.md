# (Image Captioning for Visually Impaired Individuals)

# Problem Statement

Image captioning for the visually impaired is a significant issue that needs attention. It involves generating descriptions for images understandably and accurately. This task is crucial to aid visually impaired individuals in interpreting and understanding visual content. Despite advancements in technology, creating precise and contextually appropriate captions remains a challenging task. Our project aims to address this problem and develop an efficient solution that can generate accurate and meaningful captions for images, thereby enhancing the experience for visually impaired individuals.

The main focus of this project is to generate an interpretable and meaningful set of captions for real-life images. We have also converted the generated caption to audio for the visually impaired to listen to the generated captions.

# Motivation and Experiments Performed Before the Final Solution

### 1. General Method Towards the problem

The Encoder-Decoder architecture is the most general/common way we could think of to solve this problem. Here we use a CNN model to generate the embeddings from the image which are further used as the hidden state or as the first word token for the decoder RNN model.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/a0f11881-ddad-4c33-8f7b-39361fa3d24f/Untitled.png)

The major problem with this approach is that as the length of the caption for an image becomes longer the decoder RNN models start to fail.  It is known that RNNs are not good for capturing relations in sentences with long lengths due to the problem of vanishing gradient. Therefore the last RNN blocks have the least info about the image as it is being used as a hidden state or the first word token at the first block of RNN.

Colab Link:- https://colab.research.google.com/drive/1Y2WpTAs0kEmaTFmw55XzTFio7d7p2On7?usp=sharing 

### 2. Improved Approach (Our implementation - **Merge Architecture)**

Here we introduce a change in architecture that we thought of. Instead of treating the architecture as an Encoder and Decoder, we thought of doing the work of CNN and RNN in parallel and then adding the image and text embeddings that we get from CNN and RNN respectively then passing it through the Linear layer to generate the caption.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/f9d1a1e2-5925-4449-90c3-3ee9351e710c/Untitled.png)

The method is trained on the next word prediction task to generate the captions i.e., given n-1 words for a caption we try to predict the nth word.

This method solves the problem of image importance being not propagated through time to the last RNN block. Now we have an embedding that is richer than the previous embedding since it captures the details of the image with every part of the newly formed embedding.

### Results of the above Architecture

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/f031fbc9-5235-42f4-b80e-6ff9f436131b/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/e56b65a1-cb88-47a6-93c0-8775aa73407d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/486eb1fe-7bf3-4d76-93f7-261029db446d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/842e12e5-ee4d-4147-905f-2a4f3591d331/Untitled.png)

### 3. Use of VAE

Now to improve the above-defined architecture we tried to use a better image representation (image embedding) for which we took the help of VAE to generate better image representation.

So, to get the image embeddings or (latent embeddings) for image captioning we trained the VAE as our encoder-CNN and its reparameterization trick allows us to get better embeddings because of the continuous nature of its latent space. We used these embeddings and added them to the text embeddings to get total embeddings from which we generated our captions of the image.

Colab Link :- https://colab.research.google.com/drive/1I78MTlMYQP33sn0F2HglFSw-28HREYZT?usp=sharing 

![qwerty.drawio (1).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/a296498f-0c5a-49a8-8311-36941ec0b9f5/qwerty.drawio_(1).png)

### 4. Use of GAN’s Intuition

As soon as we think of a generation task, the first thing that pops up in our minds is GANs or (Generative Adversarial Networks). Hence we tried to make a generator model that would help us do the same task by just taking the image and generating labels. But, we didn’t quite succeed here since the discriminator overpowered every generator model that we could think of.

### 5. Motivation

The previously presented solutions provide insight into generating meaningful captions. In the approaches defined above we are trying to generate captions based on how good the image embedding or the text embedding were. Let’s take a moment and try to think about what defines a good caption. A good caption is one whose text embedding is as close to the image embedding as possible, i.e., the cosine similarity between the embeddings is close to one.

This led us to think about why we did not try to maximize this cosine similarity between the text embedding and the image embedding. This takes us to our solution as stated in the following section. 

# Solution following the Motivation

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/88c7b799-d34c-4755-acbf-4819400ddbec/Untitled.png)

The above figure describes what we were trying to convey through our motivation. Let‘s say we have the image embedding (Image-1) and text embedding (Embed-1) as shown in the above figure and respectively for other data points as well. Now to predict good captions the aim of our model should be to get values of diagonal elements as close to one as possible and other elements of the diagonal matrix to be as close to zero as possible.

This gave us an idea to implement a model which gives the similarity matrix as defined above. To calculate the loss we have used the cross-entropy loss of all rows and columns of the similarity matrix with a zeros tensor containing one only at those specific row or column index respectively. Then we have backpropagated the loss to train the model.
