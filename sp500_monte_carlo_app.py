import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set the title of the app
st.title("S&P 500 Monte Carlo Simulation")

# Load the S&P 500 dataset
@st.cache_data  # Cache the data to improve performance
def load_data():
    # Replace with the path to your CSV file
    data = pd.read_csv('S&P 500 Historical Data.csv', parse_dates=['Date'])
    # Ensure the 'Price' column is numeric
    data['Price'] = data['Price'].replace('[\$,]', '', regex=True).astype(float)
    return data

data = load_data()

# Display the raw data
st.subheader("Raw Data")
st.write(data)

# Calculate daily returns
data['Daily Return'] = data['Price'].pct_change()

# Monte Carlo Simulation
st.subheader("Monte Carlo Simulation")

# Input parameters
st.sidebar.header("Simulation Parameters")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000)
num_days = st.sidebar.slider("Number of Days to Predict", 30, 365, 30)

# Calculate historical drift and volatility
mu = data['Daily Return'].mean()  # Drift
sigma = data['Daily Return'].std()  # Volatility

# Monte Carlo simulation function
def monte_carlo_simulation(start_price, num_days, num_simulations, mu, sigma):
    results = np.zeros((num_days + 1, num_simulations))
    results[0] = start_price
    for t in range(1, num_days + 1):
        shock = np.random.normal(mu, sigma, num_simulations)
        results[t] = results[t - 1] * (1 + shock)
    return results

# Run the simulation
start_price = data['Price'].iloc[-1]  # Most recent price
simulation_results = monte_carlo_simulation(start_price, num_days, num_simulations, mu, sigma)

# Plot the simulation results
st.write(f"Monte Carlo Simulation Results ({num_simulations} simulations, {num_days} days)")
plt.figure(figsize=(10, 6))
plt.plot(simulation_results)
plt.title(f"Monte Carlo Simulation of S&P 500 Prices (Next {num_days} Days)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)
st.pyplot(plt)

# Analyze the results
final_prices = simulation_results[-1]
mean_final_price = final_prices.mean()
std_final_price = final_prices.std()

st.subheader("Simulation Statistics")
st.write(f"Mean Final Price after {num_days} days: ${mean_final_price:.2f}")
st.write(f"Standard Deviation of Final Price: ${std_final_price:.2f}")

# Probability of the price exceeding a certain level
st.sidebar.header("Probability Analysis")
target_price = st.sidebar.number_input("Enter Target Price", value=start_price * 1.1)
probability = (final_prices > target_price).mean()
st.write(f"Probability of Price Exceeding ${target_price:.2f}: {probability * 100:.2f}%")

# Histogram of final prices
st.subheader("Distribution of Final Prices")
plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=50, edgecolor='black')
plt.axvline(mean_final_price, color='red', linestyle='dashed', linewidth=2, label=f"Mean Price: ${mean_final_price:.2f}")
plt.axvline(target_price, color='green', linestyle='dashed', linewidth=2, label=f"Target Price: ${target_price:.2f}")
plt.title("Distribution of Simulated Final Prices")
plt.xlabel("Final Price")
plt.ylabel("Frequency")
plt.legend()
st.pyplot(plt)