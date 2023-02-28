# Fin-Data-Denoising
Finance data denoising with MA, EMA, bilateral filter, AE(CNN Stacked, LSTM Stacked)

This project is focusing on denosing financial time series data and finding out which denoising method has good performance. The main idea is from [this blog post](https://www.qraftec.com/insights-korean/2019/3/6/deep-time-series-denosier). Implementing the result of the post is the goal of this project.

Mid-Result
- [AutoEncoder Result](./autoencoder-test.ipynb)
- [Traditional Denoising Method Result](./trad-denoising.ipynb)

The Project is ongoing.

Traditional filters([tsfilt](./tsfilt/)) is from [this repository](https://github.com/statefb/ts-spatial-filter).

- I am going to implement both CNN, LSTM stacked autoencoder to check whether the performance of CNN stacked autoencoder is better than LSTM stacked autoencoder.
- The reason why I chosed bilateral filter instead of Gaussian filter is because complex financial data noise is not expected to form a Gaussian distribution.
