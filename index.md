# Deep Learning

## Mathematics for Deep Learning

My [Mathematics for Deep Learning](./mma-notes/MathematicsForDeepLearning.nb.pdf) write-up is intended to document &mdash; with easy-to-follow but rigorous steps &mdash; multi-variable calculus as it is  applied to backpropagation. The final result for the gradients with respect to the neural net's parameters is surprisingly simple. To be fair, the apparent simplicity depends on some sweet notation.

TODO: *To make the general result more intuitive, I could add a simple but realistic two-layer neural net example at the end of the notes.*

ANOTHER TODO: *I could add more motivation for the multi-variable calculus result that was required for the derivation, and even for the ordinary chain rule, in which case the above write-up would have no prerequisites other than very good high school math.*

## References

While doing the Term 6 Independent Study with Hexi, we got a lot of our understanding of deep learning from Chapters 18 and 19 of Joel Grus, *Data Science from Scratch, 2nd Edition.* Grus has
a chapter on natural language embeddings, but he does not cover transformers, so we need another reference.

Following up on a recommendation for further reading given at the end of Chapter 19 of Grus, our next reference might be *[Deep Learning with Python, Third Edition](https://www.manning.com/books/deep-learning-with-python-third-edition)* (forthcoming) by Chollet &amp; Watson. Chollet &amp; Watson use Keras as something that you might think of as a FIL (framework insulation layer). Keras requires either TensorFlow or PyTorch as a backend. However, I don't see a compelling need to have a FIL (at this point in our learning process), and in fact, the extra noise created by mating Keras to a specific FIL seems unneccessary (unless you already preparing to deploy to a cluster).

Somehow a book by Sebastian Raschka came to my attention, he prefers to use PyTorch directly, and his introduction to LLMs is well-reviewed. Therefore, *[Build a Large Language Model (from Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)* by Raschka seems more compelling than Chollet &amp; Watson as our next reading.

## Raschka Notes

* [Envs and Resources](./envs_and_resources-raschka.html) for Raschka
* Chapter 1 and Appendix A: [Overview and Setup](./raschka/rk_ch01_and_appa-overview_and_setup.py)
* Chapter 2: [Working with Text](./raschka/rk_ch02-working_with_text.py)
* Chapter 3 (NOT YET STARTED): [Attention Mechanisms](./raschka/rk_ch03-attention_mechanisms.py)
