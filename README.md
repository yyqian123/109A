# 109A-Final-Project

## Exploiting Neural Network’s Intelligence for Quantitative Trading

### Problem Statement
Historically, classical Time Series Modelling techniques e.g. ARIMA and GARCH were go-to machine learning tools employed by Quantitative Researchers to perform price predictions of financial instruments. Recently, their utility has been slowly diminishing due to their widespread use. The field is turning toward deep learning to improve market predictions and trading strategies. Overall, the past decade has seen many advancements in deep learning that fundamentally transformed all aspects of modern life. Neural networks also have shown a lot of potential in price predictions for stock trading. In this project, we attempt to model and predict prices of financial securities (including but not limited to stock indices and single stocks) with deep learning algorithms.

### Data Resources 
We will be using panda io reader, which has the capability to pull data from various remote data sources including Yahoo Finance and Nasdaq. 
	https://pandas-datareader.readthedocs.io/en/latest/

We also have access to Bloomberg Market Data. 

### Higher Level Project Goals 
Use day closing and/or intraday prices of financial instruments to train various kinds of Neural Networks and use their intelligence for price prediction. 

Explore different neural net architectures including but are not limited to:
Long Short-Term Memory Model (LSTM); 
Multilayer Perceptron (MLP), Artificial Neural Network (ANN); 
Probabilistic Neural Network (PNN). 

Use predictions generated from our models to develop trading strategies, and subsequently, benchmark returns generated by our strategies against simple buy and hold strategy and try to outperform that.

Acquire end of the day closing price and intraday prices for stock indices and publicly listed stocks from Yahoo Finance and Bloomberg.

Perform data clean up to removal anomalies and make it consistent.

Build and train various neural networks to use them for price prediction.

Benchmark returns generated by these predictive models against buy and hold strategy to identify the best neural network technique for price prediction.

### References
http://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf

https://link.springer.com/chapter/10.1007/978-3-642-23878-9_51

https://www.sciencedirect.com/science/article/pii/S1877050917318252

http://dspace.unive.it/bitstream/handle/10579/12450/842777-1212885.pdf?sequence=2

https://www.researchgate.net/publication/316848946_An_Artificial_Neural_Network-based_Stock_Trading_System_Using_Technical_Analysis_and_Big_Data_Framework

https://sceweb.uhcl.edu/boetticher/ML_DataMining/embedding-technical-analysis-into.pdf

