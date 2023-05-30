# Fin-Data-Denoising
> Finance data denoising with MA, EMA, bilateral filter, Convolutional Autoencoder

This project is focusing on denosing financial time series data and finding out which denoising method has better performance. Finally finding new insights from noise distribution of the time series data.

Mid-Result
- [Convolutional Autoencoder Result](./autoencoder-test.ipynb)
- [Convolutional Autoencoder Traing Visualization](./results/CNN-kernel51.gif)
- [Traditional Denoising Method Result](./trad-denoising.ipynb)
- [Data Analysis: KL-divergence, KDE, Violin Plot, etc.](./noise-analysis.ipynb)

The Project is ongoing.

Traditional filters([tsfilt](./tsfilt/)) is from [this repository](https://github.com/statefb/ts-spatial-filter).

> Note.\
> The reason why I chosed bilateral filter instead of Gaussian filter is because complex financial data noise is not expected to form a Gaussian distribution.
