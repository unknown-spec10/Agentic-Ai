from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.groq import Groq

# Define the model
model = Groq(id="llama-3.3-70b-versatile", api_key="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")

# Define the financial data retrieval agent
financial_agent = Agent(
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    model=model,
    description="Retrieve and analyze financial data.",
    instructions=["Use tables to display the data."]
)

# Define the news retrieval agent
news_agent = Agent(
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    model=model,
    role="Search the web for relevant news articles."
)

# Define the multi-agent
multi_ai_agent = Agent(
    team=[financial_agent, news_agent],
    instructions=["Always include the sources", "Use tables to display the data"],
    model=model,
    show_tool_calls=True,
    markdown=True
)

# Function to get stock information
def get_stock_info(company_name):
    query = f"Share the {company_name} stock price and analyst recommendations"
    response = financial_agent.print_response(query, markdown=True)
    return response

# Function to get news articles
def get_news(company_name):
    query = f"Latest news about {company_name}"
    response = news_agent.print_response(query, markdown=True)
    return response

# Function to summarize analyst recommendations and share the latest news
def summarize_and_share_news(company_name):
    query = f"Summarize the analyst recommendations and share the latest news for {company_name}"
    response = multi_ai_agent.print_response(query, markdown=True)
    return response

# Example usage
company_name = input("Enter the stock name of the company (e.g., APPLE, NVDA): ")
#print(get_stock_info(company_name))
#print(get_news(company_name))
print(summarize_and_share_news(company_name))
