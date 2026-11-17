"""Improved fundamental analysis agent."""
from typing import List
from langchain_core.tools import BaseTool
from tools.enhanced_fundamental_tool import EnhancedFundamentalTool
from agents.base_agent import BaseAgent
from config.settings import APIConfig
from core.logging_config import setup_logging

logger = setup_logging()


# Note: this class carried a Traceloop `@agent(name="fundamental_agent")`
# decorator. It is no longer needed. The OpenInference LangChain/LangGraph
# instrumentor (see core/observability.py) hooks into LangChain's callback
# system and emits an AGENT span for every agent in the supervisor graph,
# named after its LangGraph node - including this one. For code paths that
# are *not* LangChain (plain Python helpers), use the OpenInference
# decorators instead:
#
#     from core.observability import get_tracer
#     tracer = get_tracer()
#
#     @tracer.agent(name="fundamental_agent")
#     def my_helper(...): ...
#
# `tracer.chain`, `tracer.tool` and `tracer.llm` are available the same way.
class FundamentalAgent(BaseAgent):
    """Agent for fundamental analysis."""

    def __init__(self, apiConfig: APIConfig):
        super().__init__(apiConfig, "fundamental_agent")

    def get_tools(self) -> List[BaseTool]:
        """Get fundamental analysis tools."""
        return [
            EnhancedFundamentalTool()
        ]

    def get_prompt(self) -> str:
        """Get agent prompt."""
        return (
            "You are a fundamental analysis expert specializing in comprehensive stock evaluation.\n\n"
            "ANALYSIS FRAMEWORK:\n"
            "1. Financial Health Assessment:\n"
            "   - Revenue growth trends (3-5 years)\n"
            "   - Profitability metrics (margins, ROE, ROA)\n"
            "   - Balance sheet strength (debt ratios, liquidity)\n"
            "   - Cash flow analysis (operating, free cash flow)\n\n"
            "2. Valuation Analysis:\n"
            "   - P/E, P/B, P/S ratios vs industry/market\n"
            "   - PEG ratio for growth consideration\n"
            "   - DCF model with clear assumptions\n"
            "   - Comparable company analysis\n\n"
            "3. Competitive Position:\n"
            "   - Market share and competitive advantages\n"
            "   - Industry dynamics and trends\n"
            "   - Management quality and strategy\n\n"
            "4. Risk Assessment:\n"
            "   - Business risks and challenges\n"
            "   - Regulatory and market risks\n"
            "   - Financial risks\n\n"
            "INSTRUCTIONS:\n"
            "- Provide data-driven analysis with specific numbers\n"
            "- Compare metrics to industry averages\n"
            "- Highlight both strengths and weaknesses\n"
            "- Include forward-looking insights\n"
            "- Conclude with investment thesis summary\n"
        )

