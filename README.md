
## Stock Price Prediction Using LSTM and Streamlit

#### Objective
The goal of this project is to develop a model for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The project will enable users to input a ticker symbol and predict the stock's future prices, while also providing visualizations for better understanding and analysis through a Streamlit web application.

#### Data Collection
- **Source**: Yahoo Finance
- **Dynamic Input**: Users can input any ticker symbol to fetch the stock data.
- **Data Range**: Historical data will be collected, typically covering several years to ensure enough data points for accurate prediction.

#### Data Preparation
1. **Remove Unwanted Columns**: After fetching the data, columns that are not necessary for prediction (like trading volume, adjusted close, etc.) will be removed.
2. **Calculate Moving Averages**:
   - **100-Day Moving Average (100MA)**: Calculated to smooth out the long-term trends.
   - **50-Day Moving Average (50MA)**: Calculated to identify shorter-term trends.
3. **Data Splitting**: The dataset will be split into training and testing sets in a 70:30 ratio to validate the model's performance.

#### Model Building
- **Architecture**: Long Short-Term Memory (LSTM) neural network, which is well-suited for time series prediction due to its ability to capture long-term dependencies.
- **Training**: The LSTM model will be trained on the historical stock price data to learn patterns and make future predictions.

#### Prediction
- **Output**: The model will predict the future closing prices of the stock based on the historical data provided.

#### Visualization
The project will include the following visualizations to aid in understanding and analyzing the predictions:
1. **Closing Price vs. Time Chart with 100MA**: This graph will plot the historical closing prices along with the 100-day moving average.
2. **Closing Price vs. Time Chart with 100MA & 50MA**: This graph will plot the historical closing prices along with both the 100-day and 50-day moving averages.
3. **Prediction vs. Original**: This graph will compare the predicted stock prices with the actual historical prices to visualize the accuracy of the model.

#### Streamlit Web Application
- **User Interface**: A user-friendly web application will be developed using Streamlit to allow users to input ticker symbols, view predictions, and visualize data.
- **Interactive Features**: Users can interact with the application to dynamically update inputs and view real-time visualizations.

#### Tools and Technologies
- **Data Source**: Yahoo Finance API
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, TensorFlow/Keras, Matplotlib, Seaborn, Streamlit
- **Environment**: Jupyter Notebook for development; Streamlit for deployment

#### Expected Outcomes
- A functional model capable of predicting stock prices based on historical data.
- An interactive Streamlit web application for inputting ticker symbols and viewing predictions.
- Clear and informative visualizations to understand stock trends and model accuracy.

#### Future Work
- Incorporate additional features like sentiment analysis from news articles or social media.
- Enhance the model by using more complex architectures or ensemble methods.
- Improve the Streamlit application by adding more interactive features and deploying it on a cloud platform for wider accessibility.

This project aims to provide a comprehensive solution for stock price prediction using LSTM, focusing on ease of use and insightful visualizations through an interactive Streamlit application.
