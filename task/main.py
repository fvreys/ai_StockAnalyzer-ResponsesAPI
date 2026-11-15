from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

def response_contains_mcp_list_tools(response) -> bool:
    """
    Check whether the Responses API output contains an MCP list-tools step.
    Works with SDK objects, dictionaries, and string-like responses.
    """
    try:
        text = response.model_dump_json()
    except AttributeError:
        try:
            text = str(response.model_dump())
        except AttributeError:
            text = str(response)

    text = text.lower().replace("-", "_").replace(" ", "_")

    return any(
        marker in text
        for marker in (
            "mcplisttools",
            "mcp_list_tools",
            "mcp.list_tools",
            "mcp_listtools",
            # "list_tools",  # Remove if you want to avoid false positives
        )
    )


def main():
    # Free Lite LLM using OpenAI client (not Tiny LLM - typo)
    tiny_api_key=os.getenv("TINY_API_KEY")
    if not tiny_api_key:
        raise RuntimeError("TINY_API_KEY is not set.")

    tiny_base_url = "https://litellm.aks-hs-prod.int.hyperskill.org/openai/"
    client = OpenAI(
        api_key=os.getenv("TINY_API_KEY"),
        base_url=tiny_base_url
    )

    alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not alpha_vantage_api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not set.")


    # Tool calls - two tools: MCP and Code Interpreter
    mcp_tool = {
        "type": "mcp",
        "server_label": "AlphaVantage",
        "server_url": "https://mcp.alphavantage.co/mcp",
        "require_approval": "never",
        "authorization": alpha_vantage_api_key,
    }

    code_interpreter_tool = {
        "type": "code_interpreter",
        "container": {"type": "auto", "memory_limit": "4g"},
    }

    model_used = "gpt-4.1"

    analysis_instructions = """
    You are a helpful financial agent with access to market data through Alpha Vantage MCP Server.
      a. First use the native MCP tools discovery step, McpListTools, to inspect the server tools.
      b. After the tools are listed, choose the appropriate MCP tool for the requested stock analysis.
      c. Use the Python code_interpreter for analysis and calculations of stock data. Do not use any other tools.
      d. Do not answer from general knowledge.
      e. Do not claim that market data was retrieved unless it was actually retrieved through MCP.
      f. If a tool schema is needed, get the schema before calling the data tool.
      g. If the MCP tool returns an API error, rate-limit message, or unavailable data response, explain that clearly.
      h. Explain the result in simple, non-technical language.
    
    Do not claim that data was retrieved unless it was actually retrieved through the MCP tool.
    """

    analysis_prompt = """
    ## Stock Analysis Task

    ### 1. Data Retrieval
    Retrieve monthly time series data for **'AAPL'** (latest 3 months) using:
    `TOOL_CALL(tool_name="TIME_SERIES_MONTHLY", arguments={"symbol": "AAPL", "outputsize": "compact"})`
    
    ### 2. Analysis (using code_interpreter) (REQUIRED)
    - Calculate month-over-month price changes (%)
    - Identify trend direction (up/down/sideways)
    - Compute key metrics: avg closing price, volatility, volume trends
    
    ### 3. Visualization (REQUIRED)
    Generate using `code_interpreter`:
    - **Price chart**: Monthly OHLC data
    - **Volume chart**: Trading volume per month
    
    Ensure charts have clear titles, labels, and legends.
    
    ### 4. Summary
    Provide a brief trend assessment with key insights.
    """

    # OLD - stage 3
    # Using the available Alpha Vantage MCP Server tools, analyze stock data for the stock symbol AAPL.
    # 
    # Required flow:
    # 0. First list the available MCP tools.
    # 1. Retrieve monthly stock data for AAPL.
    # 2. Use Python with the code_interpreter to analyze the last 3 months.
    # 3. Use Python with the code_interpreter to:
    #     analyse the monthly closing prices; calculate month-over-month percentage returns, 
    #     average monthly return, volatility, the best month, and the worst month. 
    #     Then provide a concise explanation of the results.
    # 4. Output for a non-technical investor:
    #    - Summary of trend direction and strength
    #    - Risk level: Low, Moderate, or High
    #    - 2-3 plain-language recommendations
    # 
    # If monthly stock data cannot be retrieved because of a real provider limit or API error, 
    # say so clearly and do not pretend that the analysis was completed.


    try:
        list_tools_response = client.responses.create(
            model=model_used,
            instructions=analysis_instructions,
            input=analysis_prompt,
            tools=[mcp_tool, code_interpreter_tool],
            tool_choice="required",
        )

        if not response_contains_mcp_list_tools (list_tools_response):
            raise RuntimeError (
                "MCP configuration error: the first model response did not call McpListTools."
            )

        print(list_tools_response.output)
        # print("✅ Client works:", final_response.output)
        print("Response: ", list_tools_response.model_dump_json(indent=2))

        ## TO DO
        # Parse the response object to find the container_id and file_id
        # from the AnnotationContainerFileCitation in the message content.
        #
        # Use client.containers.files.content.retrieve() to download the file content.
        #
        # Save it locally as stock_image.png.

    except Exception as e:
        print(f"❌ Client failed:, {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
    
