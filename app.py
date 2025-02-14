import streamlit as st
from agent import get_stock_info, get_news, summarize_and_share_news, get_google_search_results

# Title of the app
st.title("Multi-Agent Stock Analysis and News App")

# Description
st.write("""
This app uses multiple AI agents to provide insights into stocks, including:
- Stock price and analyst recommendations
- Relevant news articles
- Comprehensive web search results
""")

# Input: Company name
company_name = st.text_input("Enter the company name (e.g., Apple, NVDA):")

# Analyze Stock Button
if st.button("Analyze Stock"):
    if not company_name:
        st.error("Please enter a company name.")
    else:
        st.write("### Stock Information and Recommendations")
        try:
            stock_info = get_stock_info(company_name)
            st.markdown(stock_info, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error fetching stock information: {e}")

# News Button
if st.button("Get Latest News"):
    if not company_name:
        st.error("Please enter a company name.")
    else:
        st.write("### Latest News")
        try:
            news = get_news(company_name)
            st.markdown(news, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error fetching news articles: {e}")

# Google Search Button
if st.button("Get Web Search Results"):
    if not company_name:
        st.error("Please enter a company name.")
    else:
        st.write("### Web Search Results")
        try:
            search_results = get_google_search_results(company_name)
            st.markdown(search_results, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error fetching web search results: {e}")

# Summary Button
if st.button("Summarize & Share Insights"):
    if not company_name:
        st.error("Please enter a company name.")
    else:
        st.write("### Summary of Recommendations and News")
        try:
            summary = summarize_and_share_news(company_name)
            st.markdown(summary, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error fetching summary: {e}")

#2nd commit
