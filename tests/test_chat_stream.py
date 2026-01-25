import pytest
import json
import asyncio
from fastapi import Request
from app.routers.chat_stream import _stream_response
from app.agents.baseball_agent import BaseballStatisticsAgent
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_stream_response_success():
    # Setup
    mock_agent = AsyncMock(spec=BaseballStatisticsAgent)
    
    # Mock result with async generator for answer
    async def answer_generator():
        yield "Hello"
        yield " World"
    
    mock_agent.process_query.return_value = {
        "answer": answer_generator(),
        "tool_results": [],
        "tool_calls": [],
        "data_sources": [],
        "verified": False,
    }

    mock_request = MagicMock(spec=Request)

    # Execute
    response = await _stream_response(
        request=mock_request,
        question="test",
        filters=None,
        style="markdown",
        history=[],
        agent=mock_agent
    )

    # Collect events
    events = []
    async for event in response.body_iterator:
        events.append(event)

    # Verify
    # events are dictionaries like {'event': '...', 'data': '...'}
    # Event 0: message "Hello"
    assert events[0]['event'] == 'message'
    assert json.loads(events[0]['data'])['delta'] == 'Hello'
    
    # Event 1: message " World"
    assert events[1]['event'] == 'message'
    assert json.loads(events[1]['data'])['delta'] == ' World'
    
    # Event 2: meta (tool calls etc)
    assert events[2]['event'] == 'meta'
    
    # Event 3: done
    assert events[3]['event'] == 'done'
    assert events[3]['data'] == '[DONE]'

@pytest.mark.asyncio
async def test_stream_response_error_mid_stream():
    # Setup
    mock_agent = AsyncMock(spec=BaseballStatisticsAgent)
    
    # Async generator that raises exception
    async def error_generator():
        yield "Start"
        raise ValueError("Stream failed")
    
    mock_agent.process_query.return_value = {
        "answer": error_generator(),
        # other fields optional if code handles missing keys gracefully, 
        # but mock usually needs to be consistent
        "tool_results": [],
        "tool_calls": [],
    }

    mock_request = MagicMock(spec=Request)

    # Execute
    response = await _stream_response(
        request=mock_request,
        question="test",
        filters=None,
        style="markdown",
        history=[],
        agent=mock_agent
    )

    # Collect events
    events = []
    async for event in response.body_iterator:
        events.append(event)

    # Verify
    # Event 0: message "Start"
    assert events[0]['event'] == 'message'
    
    # Event 1: error
    assert events[1]['event'] == 'error'
    error_data = json.loads(events[1]['data'])
    assert error_data['message'] == 'streaming_error'
    assert 'Stream failed' in error_data['detail']
