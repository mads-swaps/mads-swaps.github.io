August 24, 2021

<dl>
<dt>Project Team</dt>
<dd>Nicholas Miller* (nmill@umich.edu)</dd>
<dd>Sophie Deng* (sophdeng@umich.edu)</dd>
<dd>Tim Chen* (ttcchen@umich.edu)</dd>
<dt>Github Repository</dt>
<dd><a href='https://github.com/mads-swaps/swap-for-profit'>https://github.com/mads-swaps/swap-for-profit</a></dd>
<dt>This Paper</dt>
<dd><a href='https://mads-swaps.github.io/'>https://mads-swaps.github.io/</a></dd>
</dl>

\* equal contribution

# Background

This post will introduce data scientists who are interested in cryptocurrency exchange models that can be used to predict buy and sell opportunities based on several strategies with indepth descriptions and access to code that is being used to simulate a variety of models with varying performance.

## What is Forex

Forex is short for foreign currency exchange and is the trading of currencies with the goal making profits by timing the buy and sell of specific currency paris while using candlestick charts. Strategies for trading are created by looking for patterns that can be used to predict future currency exchange price flucations.  

## Candlestick Charts

A candlestick chart is the standard plot used in visualizing trading activity where a candle is represented by a box plot that visualizes 4 prices within a given period: the high, low, open and close price.  The box, or body of the candle, is colored based on if the open price is greater than the close and differently if vice versa.  In the below chart, a white candlestick means the close price is higher than the open price meaning the price is going up.  The lines coming out of the candlestick body are called "shadows" or "wicks" and represent the price spread for the given period by extending out to the high and low price.  An individual candlestick can represnt a period as short as a second to days or weeks or more.  The chart below is a 15 minute candlestick chart so each candlestick represents a 15 minute period.

![images/15min_candle.png](images/15min_candle.png)
<center><b>Figure X</b> - Here is an example 15-minute candlestick chart for the Ethereum/Bitcoin cryptocurrency exchange rate.</center>

# Data Acquisition

# Feature Engineering

# Strategies

# AWS Infrastructure

# Evaluation

# Next Steps
