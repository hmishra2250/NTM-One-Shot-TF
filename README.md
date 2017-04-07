# One Shot Learning using Memory-Augmented Neural Networks in Tensorflow. 

Written for Tensorflow v0.12. Yet to add support for Tensorflow v1*.

Tensorflow implementation of the paper *One-shot Learning with Memory-Augmented Neural Networks*. 

Current Progress of Implementation:
- [x]  Utility Functions:
  - [x] Image Handler
  - [x] Metrics (Accuracy)
  - [x] Similarities (Cosine Similarity)
- [x] LSTM Controller and Memory Unit
- [x] Batch Generators
- [x] Omniglot Tester Code
- [ ] Unsupervised Feature Learning through Autoencoders
- [ ] Cattle/New Born Recognition

The benchmark dataset is [Omniglot dataset](https://github.com/brendenlake/omniglot). All the datasets should be placed in the [`data/`](data/) folder.

Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap, *One-shot Learning with Memory-Augmented Neural Networks*, [[arXiv](http://arxiv.org/abs/1605.06065)]
