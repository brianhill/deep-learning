# Deep Learning

## Mathematics for Deep Learning

My [Mathematics for Deep Learning](./mma-notes/MathematicsForDeepLearning.nb.pdf) write-up is intended to document &mdash; with easy-to-follow but rigorous steps &mdash; multi-variable calculus as it is  applied to backpropagation. The final result for the gradients with respect to the neural net's parameters is surprisingly simple. To be fair, the apparent simplicity depends on some sweet notation.

TODO: *To make the general result more intuitive, I could add a simple but realistic two-layer neural net example at the end of the notes.*

ANOTHER TODO: *I could add more motivation for the multi-variable calculus result that was required for the derivation, and even for the ordinary chain rule, in which case this write-up would have no prerequisites other than very good high school math.*

## References and Notes

So far I have been getting a lot of my understanding of deep learning from Chapters 18 and 19 of Joel Grus, *Data Science from Scratch, 2nd Edition.* Grus does not cover transformers, so following up on a recommendation for further reading given at the end of Chapter 19, I am looking at *[Deep Learning with Python, Third Edition](https://www.manning.com/books/deep-learning-with-python-third-edition)* (forthcoming) by Chollet &amp; Watson. I am also looking at *[Build a Large Language Model (from Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)* by Sebastian Raschka, which is doing very well on Amazon, and which uses PyTorch rather than Keras.

### Raschka Notes

* [Envs and Resources](./envs_and_resources-raschka.html) for Raschka
* Chapter 1 and Appendix A: [Overview and Setup](./raschka/rk_ch01_and_appa-overview_and_setup.py)
* Chapter 2: [Working with Text](./raschka/rk_ch02-working_with_text.py)

### Chollet &amp; Watson Notes

* [Envs and Resources](./envs_and_resources-chollet_and_watson.html) for Chollet &amp; Watson
* Chapter 2: [Building Blocks](./chollet-watson/cw_ch02-building_blocks.py)
* Chapter 3: [Python Frameworks](./chollet-watson/cw_ch03-python_frameworks.py)
