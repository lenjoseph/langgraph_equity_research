import streamlit as st
import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration
API_URL = "http://localhost:8000"

# Model pricing in dollars per 1M tokens (input, output)
MODEL_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.1": (2.50, 10.00),
    "gemini-2.5-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash": (0.15, 0.60),
}


def calculate_cost_dollars(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in dollars based on model and token usage."""
    input_price, output_price = MODEL_PRICING.get(model, (0.15, 0.60))
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost


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
                timeout=240,
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

    # Metrics Section
    metrics = data.get("metrics", {})
    if metrics:
        with st.expander("📊 Performance Metrics", expanded=False):
            # Summary metrics row
            col1, col2, col3, col4, col5 = st.columns(5)

            total_latency = metrics.get("total_latency_ms", 0)
            total_tokens = metrics.get("total_tokens", {})

            # Calculate total cost across all agents
            agent_metrics = metrics.get("agents", {})
            total_cost = sum(
                calculate_cost_dollars(
                    agent_data.get("model", ""),
                    agent_data.get("tokens", {}).get("input", 0),
                    agent_data.get("tokens", {}).get("output", 0),
                )
                for agent_data in agent_metrics.values()
            )

            with col1:
                latency_display = (
                    f"{total_latency / 1000:.1f}s"
                    if total_latency >= 1000
                    else f"{total_latency:.0f}ms"
                )
                st.metric("Total Latency", latency_display)

            with col2:
                st.metric("Total Tokens", f"{total_tokens.get('total', 0):,}")

            with col3:
                st.metric("Input Tokens", f"{total_tokens.get('input', 0):,}")

            with col4:
                st.metric("Output Tokens", f"{total_tokens.get('output', 0):,}")

            with col5:
                st.metric("Total Cost", f"${total_cost:.4f}")

            st.markdown("---")

            # Agent-level metrics table
            st.markdown("**Agent Breakdown**")

            if agent_metrics:
                # Build table data
                table_data = []
                for agent_name, agent_data in agent_metrics.items():
                    latency = agent_data.get("latency_ms", 0)
                    tokens = agent_data.get("tokens", {})
                    model = agent_data.get("model", "N/A")
                    cached = agent_data.get("cached", False)

                    input_tokens = tokens.get("input", 0)
                    output_tokens = tokens.get("output", 0)
                    cost_dollars = calculate_cost_dollars(
                        model or "", input_tokens, output_tokens
                    )

                    table_data.append(
                        {
                            "Agent": agent_name.replace("_", " ").title(),
                            "Latency": (
                                f"{latency / 1000:.2f}s"
                                if latency >= 1000
                                else f"{latency:.0f}ms"
                            ),
                            "Input Tokens": input_tokens,
                            "Output Tokens": output_tokens,
                            "Total Tokens": tokens.get("total", 0),
                            "Cost ($)": cost_dollars,
                            "Model": model or "N/A",
                            "Cached": "✓" if cached else "—",
                        }
                    )

                # Sort by agent name for consistent ordering
                table_data.sort(key=lambda x: x["Agent"])

                # Display using Streamlit's native dataframe
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Agent": st.column_config.TextColumn("Agent", width="medium"),
                        "Latency": st.column_config.TextColumn(
                            "Latency", width="small"
                        ),
                        "Input Tokens": st.column_config.NumberColumn(
                            "Input Tokens", format="%d"
                        ),
                        "Output Tokens": st.column_config.NumberColumn(
                            "Output Tokens", format="%d"
                        ),
                        "Total Tokens": st.column_config.NumberColumn(
                            "Total Tokens", format="%d"
                        ),
                        "Cost ($)": st.column_config.NumberColumn(
                            "Cost ($)", format="$%.4f"
                        ),
                        "Model": st.column_config.TextColumn("Model", width="small"),
                        "Cached": st.column_config.TextColumn("Cached", width="small"),
                    },
                )

    st.divider()

    # Combined Sentiment (main result)
    st.header("Investment Thesis")
    st.markdown(data.get("combined_sentiment", "No sentiment available"))
    disclaimer = data.get(
        "disclaimer",
        "This output is research analysis for informational purposes only and is not investment advice.",
    )
    st.caption(disclaimer)

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
