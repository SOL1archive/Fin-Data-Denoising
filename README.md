# Fin-Data-Denoising
Finance data denoising with MA, EMA, bilateral filter, AE(CNN Stacked, LSTM Stacked)

This project is focusing on denosing financial time series data and finding out which denoising method has good performance. Implementing the result of the post is the goal of this project.

Mid-Result
- [CNN Stacked AutoEncoder Result](./autoencoder-test.ipynb)
- [CNN Stacked AutoEncode Traing Visualization](./results/CNN-kernel51.gif)
- [Traditional Denoising Method Result](./trad-denoising.ipynb)

The Project is ongoing.

Traditional filters([tsfilt](./tsfilt/)) is from [this repository](https://github.com/statefb/ts-spatial-filter).

- I am going to implement both CNN, LSTM stacked autoencoder to check whether the performance of CNN stacked autoencoder is better than LSTM stacked autoencoder.
- The reason why I chosed bilateral filter instead of Gaussian filter is because complex financial data noise is not expected to form a Gaussian distribution.
