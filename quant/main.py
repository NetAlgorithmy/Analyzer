
import ccxt
import pandas as pd
import numpy as np
from ta import trend,volatility,momentum

def calculate_sma(closes, period):
    sma = trend.sma_indicator(closes, window=period)
    return sma

def calculate_atr(candles, period):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    atr = volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
    return atr

def calculate_bollinger_bands(candles, period, num_std):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    closes = df['close']
    upper_band, _, lower_band =volatility.bollinger_hband_indicator(closes, window=period, window_dev=num_std, fillna=False)
    return upper_band, lower_band

def calculate_ranging_levels(historical_prices):
    highs = historical_prices["high"]
    lows = historical_prices["low"]
    buy_line = (max(highs) + min(lows)) / 2
    sell_line = buy_line
    return buy_line, sell_line

def is_downtrend(candles):

    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    closes = df['close']
    sma = calculate_sma(closes, period=50)
    return closes.iloc[-1] < sma.iloc[-1]

def is_uptrend(candles):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    closes = df['close']
    sma = calculate_sma(closes, period=50)
    return closes.iloc[-1] > sma.iloc[-1]

def identify_trend(historical_prices, timeframe):
    if is_uptrend(historical_prices):
        return "uptrend"
    elif is_downtrend(historical_prices):
        return "downtrend"
    else:
        return "ranging"

def is_ranging(candles, atr_threshold, num_std):
    candles = np.array(candles)
    atr = calculate_atr(candles, period=14)
    upper_band, lower_band = calculate_bollinger_bands(candles, period=20, num_std=num_std)

    if atr < atr_threshold and candles[-1][4] <= upper_band and candles[-1][4] >= lower_band:
        return True
    else:
        return False

def find_support_levels(historical_prices):
    lows = historical_prices["low"]
    support_levels = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            support_levels.append(lows[i])
    return support_levels

def find_resistance_levels(historical_prices):
    highs = historical_prices["high"]
    resistance_levels = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            resistance_levels.append(highs[i])
    return resistance_levels


def identify_order_blocks(historical_prices):
    closes = historical_prices["close"]
    highs = historical_prices["high"]
    lows = historical_prices["low"]

    order_blocks = []
    for i in range(1, len(closes) - 1):
        if closes[i] > closes[i - 1] and closes[i] > closes[i + 1] and highs[i] > highs[i - 1] and lows[i] > lows[
            i - 1]:
            order_blocks.append((closes[i], highs[i], lows[i]))

    return order_blocks


def calculate_higher_high(closes):
    higher_high = False
    if len(closes) >= 3:
        if closes[-1] > max(closes[:-1]):
            higher_high = True
    return higher_high

def calculate_higher_low(closes):
    higher_low = False
    if len(closes) >= 3:
        if closes[-1] > closes[-2] and closes[-2] > closes[-3]:
            higher_low = True
    return higher_low

def calculate_lower_high(closes):
    lower_high = False
    if len(closes) >= 3:
        if closes[-1] < closes[-2] and closes[-2] < closes[-3]:
            lower_high = True
    return lower_high

def calculate_lower_low(closes):
    lower_low = False
    if len(closes) >= 3:
        if closes[-1] < min(closes[:-1]):
            lower_low = True
    return lower_low

def calculate_indicators(historical_prices):
    # Calculate technical indicators
    indicators = {}

    # Calculate higher highs and higher lows
    indicators["higher_high"] = calculate_higher_high(historical_prices["high"])
    indicators["higher_low"] = calculate_higher_low(historical_prices["low"])

    # Calculate lower highs and lower lows
    indicators["lower_high"] = calculate_lower_high(historical_prices["high"])
    indicators["lower_low"] = calculate_lower_low(historical_prices["low"])

    # Calculate buy line and sell line for ranging market
    indicators["buy_line"], indicators["sell_line"] = calculate_ranging_levels(historical_prices)

    # Calculate other technical indicators as needed
    # Example:
    close_prices = pd.Series(historical_prices["close"])
    diff = close_prices.diff(1)
    indicators["RSI"] = momentum.RSIIndicator(diff).rsi()
    indicators["MACD"] = trend.MACD(diff).macd()
    indicators["price"] = historical_prices["close"][-1]

    return indicators


# Add custom stylesheet

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.core.window import Window


class MainWindow(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [10, 10, 10, 10]
        self.spacing = 10
        self.background_color = (0.95, 0.95, 0.95, 1)  # Set window background color

        self.label = Label(text='', size_hint=(1, 0.8), halign='center', valign='middle', font_size='16sp')
        self.add_widget(self.label)

        hbox = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)

        self.input = TextInput(hint_text='Symbol', size_hint=(0.7, None), height=40, multiline=False)
        self.input.background_color = (0.15, 0.15, 0.15, 1)  # Set input background color
        self.input.foreground_color = (1, 1, 1, 1)  # Set input text color
        hbox.add_widget(self.input)

        self.button = Button(text='Start Analysis', size_hint=(0.3, None), height=40)
        self.button.background_color = (0.15, 0.15, 0.15, 1)  # Set button background color
        self.button.color = (1, 1, 1, 1)  # Set button text color
        self.button.bind(on_release=self.start_analysis)
        hbox.add_widget(self.button)

        self.add_widget(hbox)

        self.exchange = ccxt.huobi()
        self.symbol = ''
        self.timeframe_indicators = '1m'
        self.timeframe_trend = '5m'
        self.timeframe_support_resistance = '1h'
        self.timeframe_order_blocks = '30m'

        self.timer = None
    def start_analysis(self, instance):
        self.symbol = self.input.text
        self.update_analysis()
        self.timer = Clock.schedule_interval(self.update_analysis, 5)


    def update_analysis(self, dt=None):
        historical_prices = {}
        timeframes = {
            self.timeframe_indicators: self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe_indicators),
            self.timeframe_trend: self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe_trend),
            self.timeframe_support_resistance: self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe_support_resistance),
            self.timeframe_order_blocks: self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe_order_blocks)
        }
        for timeframe, candles in timeframes.items():
            historical_price = {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
            for candle in candles:
                historical_price["timestamp"].append(candle[0])
                historical_price["open"].append(candle[1])
                historical_price["high"].append(candle[2])
                historical_price["low"].append(candle[3])
                historical_price["close"].append(candle[4])
                historical_price["volume"].append(candle[5])
            historical_prices[timeframe] = historical_price

        technical_indicators = calculate_indicators(historical_prices[self.timeframe_indicators])
        trend = identify_trend(historical_prices[self.timeframe_trend], timeframe=self.timeframe_indicators)
        support_levels = find_support_levels(historical_prices[self.timeframe_support_resistance])
        resistance_levels = find_resistance_levels(historical_prices[self.timeframe_support_resistance])
        order_blocks = identify_order_blocks(historical_prices[self.timeframe_order_blocks])
        rsi_series = technical_indicators['RSI']
        rsi_value = rsi_series.iloc[-1]  # Get the latest value from the Series
        if not pd.isna(rsi_value):
            if rsi_value <= 30:
                technical_indicators['oversold'] = True

            elif rsi_value >= 70:
                technical_indicators['overbought'] = True
                technical_indicators['oversold'] = False
            else:
                technical_indicators['overbought'] = False
                technical_indicators['oversold'] = False
        else:
            technical_indicators['overbought'] = False
            technical_indicators['oversold'] = False

        if 'MACD' in technical_indicators:
            macd_series = technical_indicators['MACD']
            macd_value = macd_series.iloc[-1]  # Get the latest value from the Series
            if not pd.isna(macd_value):
                if macd_value > 0:
                    technical_indicators['bullish_macd'] = True
                elif macd_value < 0:
                    technical_indicators['bearish_macd'] = True
        technical_indicators.pop('RSI', None)
        technical_indicators.pop('MACD', None)
        # Perform analysis and update the label text
        analysis = "=== Market Analysis ===\n"
        analysis += f"Trend: {trend}\n"
        analysis += f"Support Levels: {support_levels[-1]}\n"
        analysis += f"Resistance Levels: {resistance_levels[-1]}\n"
        analysis += f"Order Blocks: {order_blocks[-1]}\n"

        if trend != "ranging":
            technical_indicators.pop('buy_line', None)
            technical_indicators.pop('sell_line', None)
        if trend != 'uptrend':
            technical_indicators.pop('higher_high', None)
            technical_indicators.pop('higher_low', None)
        if trend != 'downtrend':
            technical_indicators.pop('lower_high', None)
            technical_indicators.pop('lower_low', None)
        technical_indicators.pop('RSI', None)
        technical_indicators.pop('MACD', None)
        analysis += "==========================\n"
        analysis += f"Technical Indicators: \n"
        analysis+=f'{technical_indicators} \n'
        analysis += "==========================\n"
        if technical_indicators and trend and support_levels and resistance_levels and order_blocks:
            if trend == "uptrend":
                if technical_indicators.get("higher_high"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Higher high detected in an uptrend.\n"
                elif technical_indicators.get("higher_low"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Higher low detected in an uptrend.\n"
                else:
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "No specific trading signal detected in the uptrend.\n"
            elif trend == "downtrend":
                if technical_indicators.get("lower_high"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Lower high detected in a downtrend.\n"
                elif technical_indicators.get("lower_low"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Lower low detected in a downtrend.\n"
                else:
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "No specific trading signal detected in the downtrend.\n"
            if trend == "uptrend":
                if technical_indicators.get("oversold"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Oversold condition detected in an uptrend. Consider entering a long position.\n"
                elif technical_indicators.get("overbought"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Overbought condition detected in an uptrend. Consider entering a short position.\n"
                else:
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "No specific trading signal detected in the uptrend.\n"
            elif trend == "downtrend":
                if technical_indicators.get("oversold"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Oversold condition detected in a downtrend. Consider entering a short position.\n"
                elif technical_indicators.get("overbought"):
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Overbought condition detected in a downtrend. Consider entering a long position.\n"
                else:
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "No specific trading signal detected in the downtrend.\n"
            elif trend == "ranging":
                if technical_indicators["price"] > technical_indicators["sell_line"]:
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Price above the sell line in a ranging market.\n"
                elif technical_indicators["price"] < technical_indicators["buy_line"]:
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "Price below the buy line in a ranging market.\n"
                else:
                    analysis += "=== Trading Suggestion ===\n"
                    analysis += "No specific trading signal detected in the ranging market.\n"

        self.label.text = analysis

        def on_stop(self):
            if self.timer:
                self.timer.cancel()

class MarketAnalysisApp(App):
    def build(self):
        self.title = 'Real-time Market Analysis'
        return MainWindow()

if __name__ == '__main__':
    MarketAnalysisApp().run()