import streamlit as st
import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Equity Research Agent",
    page_icon=None,
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .sentiment-bullish { color: #00d26a; font-weight: bold; }
    .sentiment-bearish { color: #ff6b6b; font-weight: bold; }
    .sentiment-neutral { color: #ffd93d; font-weight: bold; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("AI Equity Research Agent")
st.markdown("*Multi-agent equity analysis powered by LangGraph*")

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = None

# Main area input form
st.header("Research Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, GOOGL",
    ).upper()

with col2:
    trade_direction = st.selectbox(
        "Trade Direction",
        options=["long", "short"],
        index=0,
    )

with col3:
    trade_duration = st.selectbox(
        "Trade Duration",
        options=["day_trade", "swing_trade", "position_trade"],
        index=2,
        format_func=lambda x: x.replace("_", " ").title(),
    )

analyze_button = st.button("Analyze", type="primary")

st.divider()

# Main content
if analyze_button:
    if not ticker:
        st.error("Please enter a stock ticker")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def make_api_request():
            """Make the API request in a separate thread."""
            return requests.post(
                f"{API_URL}/research-equity",
                json={
                    "ticker": ticker,
                    "trade_direction": trade_direction,
                    "trade_duration": trade_duration,
                },
                timeout=120,
            )

        # Countdown intervals in seconds
        countdown_messages = [
            (0, "~1 minute remaining"),
            (15, "~45 seconds remaining"),
            (30, "~30 seconds remaining"),
            (45, "~15 seconds remaining"),
        ]

        status_text.info(f"Analyzing {ticker}... ~1 minute remaining")

        try:
            # Run API request in background thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(make_api_request)
                start_time = time.time()

                # Update countdown while waiting for response
                while not future.done():
                    elapsed = time.time() - start_time

                    # Find appropriate countdown message
                    current_message = countdown_messages[0][1]
                    for threshold, message in countdown_messages:
                        if elapsed >= threshold:
                            current_message = message

                    # Update progress bar (cap at 90% until complete)
                    progress = min(int((elapsed / 60) * 90), 90)
                    progress_bar.progress(progress)
                    status_text.info(f"Analyzing {ticker}... {current_message}")

                    time.sleep(1)

                # Get the response
                response = future.result()

            progress_bar.progress(100)
            status_text.empty()

            if response.status_code == 200:
                data = response.json()
                st.session_state.results = data

            elif response.status_code == 400:
                st.error(f"Invalid ticker: {ticker}")
            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to API. Make sure the server is running on localhost:8000"
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The analysis is taking too long.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display results if available
if st.session_state.results:
    data = st.session_state.results

    # Success message
    st.success(f"Analysis complete for {data.get('ticker', ticker)}")

    # Combined Sentiment (main result)
    st.header("Investment Thesis")
    st.markdown(data.get("combined_sentiment", "No sentiment available"))

    st.divider()

    # Individual Agent Results
    st.header("Detailed Analysis")

    sentiment_data = data.get("sentiment_analysis", {})

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Fundamental Analysis", expanded=True):
            st.markdown(sentiment_data.get("fundamental", "Not available"))

        with st.expander("Technical Analysis", expanded=True):
            st.markdown(sentiment_data.get("technical", "Not available"))

        with st.expander("Macro Analysis", expanded=True):
            st.markdown(sentiment_data.get("macro", "Not available"))

    with col2:
        with st.expander("Industry Analysis", expanded=True):
            st.markdown(sentiment_data.get("industry", "Not available"))

        with st.expander("Peer Analysis", expanded=True):
            st.markdown(sentiment_data.get("peer", "Not available"))

        with st.expander("Headline Analysis", expanded=True):
            st.markdown(sentiment_data.get("headline", "Not available"))

        with st.expander("SEC Filings Analysis", expanded=True):
            st.markdown(sentiment_data.get("filings", "Not available"))
