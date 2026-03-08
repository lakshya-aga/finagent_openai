import asyncio
from agent_workflow import run_workflow, WorkflowInput

async def main():

    prompt = """
    Build a momentum trading strategy on S&P500 stocks.

    Steps:
    - fetch daily prices
    - compute zscore of average returns of last 30 days
    - construct long-short portfolio with top 50 stocks longed and bottom 50 shorted
    - produce asset_returns and asset_weights
    - conduct backtest
    """

    complex_prompt = """
    Build a research notebook that uses our internal data and ML libraries to study an event-based directional
    prediction pipeline on SPY. Fetch the data, construct dollar bars, detect events with a CUSUM filter, label 
    them with the triple barrier method, create a compact but sensible feature set, train a baseline classifier,
    and evaluate the results using time-based train/test splits. Show intermediate outputs and keep the notebook
    readable and modular.
    """

    result = await run_workflow(
        WorkflowInput(
            input_as_text=complex_prompt
        )
    )

    print("\n\nFINAL RESULT")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())