from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.groq import Groq
from agno.models.deepseek import DeepSeek
from agno.tools.googlesearch import GoogleSearch


# Define the model
model = Groq(id="llama-3.3-70b-versatile", api_key="sk-a8338be3d2c34ed28ebcec1068414e08")

# Define the financial data retrieval agent
financial_agent = Agent(
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    model=model,
    description="Retrieve and analyze financial data.",
    #instructions=["Use tables to display the data."]
)
# Define the Google search agent
google_search_agent = Agent(
    tools=[GoogleSearch()],
    show_tool_calls=True,
    model=model,
    instructions=['Display the data in a organized manner'],
    role="Search the web for comprehensive information."
)

# Define the news retrieval agent
news_agent = Agent(
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    model=model,
    role="Search the web for relevant news articles about."
)

# Define the multi-agent
multi_ai_agent = Agent(
    team=[financial_agent, news_agent,google_search_agent],
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

#Function to google search
def get_google_search_results(company_name):
    query = f" Search relevent Stock related info about {company_name}"
    response = google_search_agent.print_response(query, markdown=True)
    return response

# Function to summarize analyst recommendations and share the latest news
def summarize_and_share_news(company_name):
    query = f"Summarize the analyst recommendations and share the latest news for {company_name}"
    response = multi_ai_agent.print_response(query, markdown=True)
    return response

# Example usage
company_name = input("Enter the stock name of the company (e.g., APPLE, NVDA): ")
print(get_stock_info(company_name))
#print(get_news(company_name))
print(summarize_and_share_news(company_name))
print(get_google_search_results(company_name))
