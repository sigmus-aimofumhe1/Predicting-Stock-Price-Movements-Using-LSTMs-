This project focuses on predicting stock prices using advanced LSTM-based deep learning models. 
Historical stock data was collected from **Yahoo Finance** for Apple (AAPL) and Tesla (TSLA) using the `yfinance` library. 
The data was then preprocessed through exploratory data analysis (EDA), scaling, and a sliding window technique to convert it into time-series sequences suitable for model training. 
The first model developed was a baseline LSTM with a single hidden layer, serving as a benchmark for later improvements.

In the later stages, I implemented more complex architectures such as a Stacked LSTM and a Bidirectional LSTM to capture deeper temporal dependencies in stock trends. 
These models significantly improved prediction accuracy, as shown by reduced Mean Squared Error (MSE) and Mean Absolute Error (MAE) values. 
Visualization of actual versus predicted prices demonstrated that the Bidirectional LSTM performed best. 
All trained models were saved for further refinement in the next phase, where I plan to integrate real-time stock data and a simple web-based dashboard for live visualization.
