
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


# This class is a sample. Feel free to customize it.
class BBRSI(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy intervace version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # minimal_roi = {
    #     # "40": 0.0, # Sell when trade is non-loosing (in effect after 40 minutes)
    #     # "30": 0.01, # Sell when 1% profit was reached (in effect after 30 minutes)
    #     # "20": 0.02, # Sell when 2% profit was reached (in effect after 20 minutes)
    #     # "0": 0.05 # Sell whenever 4% profit was reached (in effect immediately)
    #     "0": 100
    # }
    minimal_roi = {
        "125": 0, # after 125 minutes, exit if I make 0 profit
        "74": 0.03572628005452015, # after 74 minutes, exit if I make 3.5% profit
        "30": 0.08750273438950948, # after 30 min, exit if I make 8.7% profit
        "0": 0.13604455019472522 # after 0 min, exit if I make 13.6% profit
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # stoploss = -0.10
    stoploss = -0.38357416420491

    # trailing stoploss
    trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal ticker interval for the strategy
    ticker_interval = '15m'

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicator
        # ------------------------------------

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        """
        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # Awesome oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # Commodity Channel Index: values Oversold:<-100, Overbought:>100
        dataframe['cci'] = ta.CCI(dataframe)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # Plus Directional Indicator / Movement
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # ROC
        dataframe['roc'] = ta.ROC(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # Stoch
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # Stoch RSI
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']
        """

        # Overlap Studies
        # ------------------------------------

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        """
        # EMA - Exponential Moving Average
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        """



        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        """
        # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # Dragonfly Doji: values [0, 100]
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # Morningstar: values [0, 100]
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # Three White Soldiers: values [0, 100]
        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]
        """

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        """
        # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # Gravestone Doji: values [0, 100]
        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # Dark Cloud Cover: values [0, 100]
        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # Evening Doji Star: values [0, 100]
        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # Evening Star: values [0, 100]
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
        """

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        """
        # Three Line Strike: values [0, -100, 100]
        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # Spinning Top: values [0, -100, 100]
        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # Engulfing: values [0, -100, 100]
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # Harami: values [0, -100, 100]
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # Three Outside Up/Down: values [0, -100, 100]
        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]
        """

        # Chart type
        # ------------------------------------
        """
        # Heikinashi stategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        """

        # Retrieve best bid and best ask
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # If RSI is above 20, that means people are still buying
        # If the closing price is less than the lower4 bb, that means the price is relatively quite low
        dataframe.loc[
            (
                (dataframe['rsi'] < 20) &
                (dataframe['close'] < dataframe['bb_lowerband'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # If the closing price is greater than the upper band, that means the price is relatively high
        dataframe.loc[
            (
                (dataframe['rsi'] > 37) &
                (dataframe['close'] > dataframe['bb_upperband'])
            ),
            'sell'] = 1
        return dataframe
