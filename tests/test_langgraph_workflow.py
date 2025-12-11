"""
Test LangGraph workflow integration.
Ensures router -> planner -> retriever -> answerer flow works.
"""

import sys
import os

# Change to agents directory to avoid circular import issues
agents_dir = os.path.join(os.path.dirname(__file__), '../agents')
os.chdir(agents_dir)
sys.path.insert(0, agents_dir)

from agent_state import build_assistant_graph, AgentState


def test_graph_builds():
    """Test that graph compiles without errors."""
    print("\n[TEST 1] Graph Compilation")
    
    try:
        graph = build_assistant_graph()
        assert graph is not None
        print("[PASS] Graph compiled successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Graph compilation failed: {e}")
        raise


def test_simple_product_query():
    """Test basic product search query."""
    print("\n[TEST 2] Simple Product Query")
    
    graph = build_assistant_graph()
    
    initial_state: AgentState = {
        "user_query": "disposable paper plates",
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    
    final_state = graph.invoke(initial_state)
    
    # Assertions
    assert final_state.get("final_response"), "No final response"
    assert final_state.get("tts_summary"), "No TTS summary"
    assert len(final_state.get("retrieved_products", [])) > 0, "No products retrieved"
    assert len(final_state.get("agent_logs", [])) >= 4, "Missing agent logs (expected >=4)"
    assert final_state.get("intent"), "No intent extracted"
    
    print(f"[PASS] Retrieved {len(final_state['retrieved_products'])} products")
    print(f"[PASS] Generated response: {final_state['final_response'][:100]}...")
    return True


def test_intent_with_constraints():
    """Test router extracts budget constraints."""
    print("\n[TEST 3] Intent Extraction with Constraints")
    
    graph = build_assistant_graph()
    
    initial_state: AgentState = {
        "user_query": "eco-friendly stainless steel cleaner under fifteen dollars",
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    
    final_state = graph.invoke(initial_state)
    
    intent = final_state.get("intent")
    assert intent, "Intent not extracted"
    
    constraints = intent.constraints
    assert constraints.budget_max is not None, "Budget constraint not extracted"
    assert constraints.budget_max <= 15.5, f"Budget incorrect: {constraints.budget_max}"
    assert constraints.eco_friendly is True, "Eco-friendly not detected"
    
    print(f"[PASS] Budget max: ${constraints.budget_max}")
    print(f"[PASS] Eco-friendly: {constraints.eco_friendly}")
    return True


def test_web_search_trigger():
    """Test planner triggers web search for 'current' queries."""
    print("\n[TEST 4] Web Search Trigger")
    
    graph = build_assistant_graph()
    
    initial_state: AgentState = {
        "user_query": "current price of steel cleaner",
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    
    final_state = graph.invoke(initial_state)
    
    plan = final_state.get("retrieval_plan")
    assert plan, "No retrieval plan"
    assert plan.data_sources.use_live is True, "Web search not triggered"
    
    # Check tool calls
    tool_calls = final_state.get("tool_calls", [])
    web_calls = [t for t in tool_calls if t.tool_name == "web.search"]
    assert len(web_calls) > 0, "web.search not called"
    
    print(f"[PASS] Web search triggered: {plan.data_sources.use_live}")
    print(f"[PASS] web.search called {len(web_calls)} time(s)")
    return True


def test_citations_present():
    """Test that citations are generated."""
    print("\n[TEST 5] Citations Generation")
    
    graph = build_assistant_graph()
    
    initial_state: AgentState = {
        "user_query": "disposable plates",
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    
    final_state = graph.invoke(initial_state)
    
    citations = final_state.get("citations", [])
    assert len(citations) > 0, "No citations generated"
    
    # Check citation structure
    cite = citations[0]
    assert cite.source in ["private", "web"], f"Invalid source: {cite.source}"
    assert cite.doc_id or cite.url, "Citation missing doc_id/url"
    
    print(f"[PASS] {len(citations)} citations generated")
    print(f"[PASS] Sample citation: {cite.source} - {cite.doc_id or cite.url}")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LANGGRAPH WORKFLOW TESTS")
    print("="*60)
    
    tests = [
        test_graph_builds,
        test_simple_product_query,
        test_intent_with_constraints,
        test_web_search_trigger,
        test_citations_present,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED!")
    print("="*60 + "\n")
    
    sys.exit(0 if failed == 0 else 1)
