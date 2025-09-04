# Open WebUI Chat Processing Refactor Plan

## Executive Summary

This document outlines the refactor plan to transform Open WebUI's chat processing from a hardcoded feature pipeline to a dynamic, LLM-driven agent system where external features (memory, web search, image generation, code interpreter, RAG) are treated as invocable tools/agents.

## Current Architecture Analysis

### Current Flow
```
Chat Request → process_chat_payload() → Fixed Feature Pipeline → process_chat_response()
```

### Current Feature Pipeline (Fixed Order)
1. **Pipeline Inlet Filter** - External pipeline processing
2. **Filter Inlet Functions** - Custom function filters  
3. **Memory Handler** - Query user memory, inject context
4. **Web Search Handler** - Generate queries, search web, add files
5. **Image Generation Handler** - Generate images from prompts
6. **Code Interpreter Setup** - Add system prompt for code execution
7. **Tools Function Calling** - Process external tools
8. **Files/RAG Handler** - Process documents, retrieve context
9. **Context Injection** - Add retrieved context to messages

### Current Implementation Locations
- **Entry Point**: [`main.py:394`](backend/open_webui/main.py:394) - `chat_completion()`
- **Core Processing**: [`main.py:495`](backend/open_webui/main.py:495) - `process_chat()`
- **Payload Processing**: [`middleware.py:753`](backend/open_webui/utils/middleware.py:753) - `process_chat_payload()`
- **Response Processing**: [`middleware.py:1072`](backend/open_webui/utils/middleware.py:1072) - `process_chat_response()`

### Current Feature Handlers
| Feature | Handler Function | Location | Input/Output |
|---------|------------------|----------|--------------|
| Memory | `chat_memory_handler()` | `middleware.py:324` | Modifies `form_data["messages"]` |
| Web Search | `chat_web_search_handler()` | `middleware.py:363` | Adds to `form_data["files"]` |
| Image Generation | `chat_image_generation_handler()` | `middleware.py:525` | Modifies `form_data["messages"]` |
| Code Interpreter | Inline setup | `middleware.py:907` | Modifies `form_data["messages"]` |
| RAG/Files | `chat_completion_files_handler()` | `middleware.py:626` | Returns sources for context |
| Tools | `chat_completion_tools_handler()` | `middleware.py:128` | Processes tool calls |

### Current Issues
1. **Fixed Pipeline Order**: Features always execute in predetermined sequence
2. **Mixed Interfaces**: Inconsistent input/output patterns across handlers
3. **No Dynamic Discovery**: LLM cannot choose which features to use
4. **Tight Coupling**: Features directly modify form_data structure
5. **Limited Composability**: Cannot combine features in flexible ways

## Target Architecture

### New Flow
```
Chat Request → Load Available Agents → Add to tools_dict → LLM Tool Selection → Agent Execution → Response Generation
```

### Agent-Based Design Principles
1. **Unified Interface**: All features implement consistent agent interface
2. **Dynamic Discovery**: LLM decides which agents to invoke
3. **Composable**: Agents can be combined in any order
4. **Extensible**: Easy to add new agents
5. **Decoupled**: Agents return structured data, don't modify form_data

### Core Interfaces

#### Base Agent Interface
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

class AgentResult(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    files: List[str] = []
    sources: List[Dict] = []

class ChatAgent(ABC):
    @abstractmethod
    async def execute(self, params: Dict[str, Any], context: 'ChatContext') -> AgentResult:
        """Execute the agent with given parameters"""
        pass
    
    @abstractmethod
    def get_tool_spec(self) -> Dict[str, Any]:
        """Return OpenAI tool specification for this agent"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for tool calling"""
        pass
```

#### Chat Context
```python
class ChatContext(BaseModel):
    user: UserModel
    request: Request
    messages: List[Dict]
    metadata: Dict[str, Any]
    app_state: Any
    event_emitter: Callable
    event_caller: Callable
```

## Agent Implementations

### 1. Memory Agent
```python
class MemoryAgent(ChatAgent):
    name = "query_memory"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "query_memory",
                "description": "Query user's memory for relevant past conversations and context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search in user's memory"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of memory items to retrieve",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        query = params.get("query", "")
        k = params.get("k", 3)
        
        # Use existing memory query logic
        results = await query_memory(
            context.request, 
            QueryMemoryForm(content=query, k=k), 
            context.user
        )
        
        user_context = ""
        if results and hasattr(results, "documents"):
            for doc_idx, doc in enumerate(results.documents[0]):
                created_at = "Unknown Date"
                if results.metadatas[0][doc_idx].get("created_at"):
                    timestamp = results.metadatas[0][doc_idx]["created_at"]
                    created_at = time.strftime("%Y-%m-%d", time.localtime(timestamp))
                user_context += f"{doc_idx + 1}. [{created_at}] {doc}\n"
        
        return AgentResult(
            content=f"Retrieved memory context:\n{user_context}",
            metadata={"memory_results": len(results.documents[0]) if results else 0}
        )
```

### 2. Web Search Agent
```python
class WebSearchAgent(ChatAgent):
    name = "search_web"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for current information and return file references for further processing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query or topic to search for"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        query = params.get("query", "")
        
        # Generate search queries using existing logic
        await context.event_emitter({
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "Generating search queries",
                "done": False,
            },
        })
        
        try:
            # Use existing query generation logic
            res = await generate_queries(
                context.request,
                {
                    "model": context.metadata.get("model"),
                    "messages": context.messages,
                    "prompt": query,
                    "type": "web_search",
                },
                context.user,
            )
            
            response = res["choices"][0]["message"]["content"]
            try:
                bracket_start = response.find("{")
                bracket_end = response.rfind("}") + 1
                if bracket_start != -1 and bracket_end != -1:
                    response = response[bracket_start:bracket_end]
                    queries = json.loads(response).get("queries", [])
                else:
                    queries = [response]
            except:
                queries = [response]
                
            if not queries or (len(queries) == 1 and queries[0].strip() == ""):
                queries = [query]
                
        except Exception as e:
            queries = [query]
        
        # Execute web search
        await context.event_emitter({
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "Searching the web",
                "done": False,
            },
        })
        
        results = await process_web_search(
            context.request,
            SearchForm(queries=queries),
            user=context.user,
        )
        
        files = []
        if results:
            if results.get("collection_names"):
                for col_idx, collection_name in enumerate(results.get("collection_names")):
                    files.append({
                        "collection_name": collection_name,
                        "name": ", ".join(queries),
                        "type": "web_search",
                        "urls": results["filenames"],
                        "queries": queries,
                    })
            elif results.get("docs"):
                files.append({
                    "docs": results["docs"],
                    "name": ", ".join(queries),
                    "type": "web_search",
                    "urls": results["filenames"],
                    "queries": queries,
                })
        
        await context.event_emitter({
            "type": "status",
            "data": {
                "action": "web_search",
                "description": f"Searched {len(results.get('filenames', []))} sites",
                "urls": results.get("filenames", []),
                "done": True,
            },
        })
        
        return AgentResult(
            content=f"Web search completed for '{query}'. Found {len(results.get('filenames', []))} sources. Use query_documents to retrieve specific information from these sources.",
            files=files,
            metadata={"urls": results.get("filenames", []), "queries": queries, "original_query": query}
        )
```

### 3. Image Generation Agent
```python
class ImageGenerationAgent(ChatAgent):
    name = "generate_image"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on a text prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt describing the image to generate"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        prompt = params.get("prompt", "")
        
        await context.event_emitter({
            "type": "status",
            "data": {"description": "Generating an image", "done": False},
        })
        
        try:
            images = await image_generations(
                request=context.request,
                form_data=GenerateImageForm(prompt=prompt),
                user=context.user,
            )
            
            await context.event_emitter({
                "type": "status",
                "data": {"description": "Generated an image", "done": True},
            })
            
            await context.event_emitter({
                "type": "files",
                "data": {
                    "files": [
                        {"type": "image", "url": image["url"]}
                        for image in images
                    ]
                },
            })
            
            return AgentResult(
                content="Image has been generated successfully",
                metadata={"images": [img["url"] for img in images]}
            )
            
        except Exception as e:
            await context.event_emitter({
                "type": "status",
                "data": {
                    "description": "An error occurred while generating an image",
                    "done": True,
                },
            })
            
            return AgentResult(
                content="Unable to generate an image, an error occurred",
                metadata={"error": str(e)}
            )
```

### 4. Code Interpreter Agent
```python
class CodeInterpreterAgent(ChatAgent):
    name = "execute_code"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute Python code and return the results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (currently only 'python' supported)",
                            "default": "python"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        code = params.get("code", "")
        language = params.get("language", "python")
        
        if language != "python":
            return AgentResult(
                content="Only Python code execution is currently supported",
                metadata={"error": "Unsupported language"}
            )
        
        try:
            # Use existing code execution logic
            if context.app_state.config.CODE_INTERPRETER_ENGINE == "pyodide":
                output = await context.event_caller({
                    "type": "execute:python",
                    "data": {
                        "id": str(uuid4()),
                        "code": code,
                        "session_id": context.metadata.get("session_id", None),
                    },
                })
            elif context.app_state.config.CODE_INTERPRETER_ENGINE == "jupyter":
                output = await execute_code_jupyter(
                    context.app_state.config.CODE_INTERPRETER_JUPYTER_URL,
                    code,
                    # ... other jupyter params
                )
            else:
                output = {"stdout": "Code interpreter engine not configured."}
            
            result_content = ""
            if isinstance(output, dict):
                stdout = output.get("stdout", "")
                stderr = output.get("stderr", "")
                result = output.get("result", "")
                
                if stdout:
                    result_content += f"Output:\n{stdout}\n"
                if stderr:
                    result_content += f"Errors:\n{stderr}\n"
                if result:
                    result_content += f"Result:\n{result}\n"
            
            return AgentResult(
                content=f"Code executed successfully:\n```python\n{code}\n```\n\n{result_content}",
                metadata={"output": output}
            )
            
        except Exception as e:
            return AgentResult(
                content=f"Code execution failed: {str(e)}",
                metadata={"error": str(e)}
            )
```

### 5. RAG Agent
```python
class RAGAgent(ChatAgent):
    name = "query_documents"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "query_documents",
                "description": "Query uploaded documents, knowledge base, and web search results for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search in documents and knowledge sources"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of relevant chunks to retrieve",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        query = params.get("query", "")
        k = params.get("k", 5)
        
        # Get files from context metadata (includes uploaded files, web search results, etc.)
        files = context.metadata.get("files", [])
        if not files:
            return AgentResult(
                content="No documents or sources available to query. Try uploading documents or using web search first.",
                metadata={"files_count": 0}
            )
        
        # Generate retrieval queries using existing logic
        try:
            queries_response = await generate_queries(
                context.request,
                {
                    "model": context.metadata.get("model"),
                    "messages": context.messages,
                    "type": "retrieval",
                },
                context.user,
            )
            queries_response = queries_response["choices"][0]["message"]["content"]
            
            try:
                bracket_start = queries_response.find("{")
                bracket_end = queries_response.rfind("}") + 1
                if bracket_start != -1 and bracket_end != -1:
                    queries_response = queries_response[bracket_start:bracket_end]
                    queries = json.loads(queries_response).get("queries", [])
                else:
                    queries = [queries_response]
            except:
                queries = [queries_response]
                
            if not queries:
                queries = [query]
        except:
            queries = [query]
        
        # Use existing RAG logic with ThreadPoolExecutor for performance
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                sources = await loop.run_in_executor(
                    executor,
                    lambda: get_sources_from_items(
                        request=context.request,
                        items=files,
                        queries=queries,
                        embedding_function=lambda q, prefix: context.app_state.EMBEDDING_FUNCTION(
                            q, prefix=prefix, user=context.user
                        ),
                        k=k,
                        reranking_function=(
                            (lambda sentences: context.app_state.RERANKING_FUNCTION(
                                sentences, user=context.user
                            )) if context.app_state.RERANKING_FUNCTION else None
                        ),
                        k_reranker=context.app_state.config.TOP_K_RERANKER,
                        r=context.app_state.config.RELEVANCE_THRESHOLD,
                        hybrid_bm25_weight=context.app_state.config.HYBRID_BM25_WEIGHT,
                        hybrid_search=context.app_state.config.ENABLE_RAG_HYBRID_SEARCH,
                        full_context=context.app_state.config.RAG_FULL_CONTEXT,
                        user=context.user,
                    ),
                )
        except Exception as e:
            return AgentResult(
                content=f"Error retrieving information: {str(e)}",
                metadata={"error": str(e), "query": query}
            )
        
        if not sources:
            return AgentResult(
                content=f"No relevant information found for query: '{query}'",
                metadata={"query": query, "sources_count": 0}
            )
        
        # Format retrieved context
        context_string = ""
        citation_idx_map = {}
        
        for source in sources:
            if "document" in source:
                for document_text, document_metadata in zip(
                    source["document"], source["metadata"]
                ):
                    source_name = source.get("source", {}).get("name", "Unknown")
                    source_id = (
                        document_metadata.get("source", None)
                        or source.get("source", {}).get("id", None)
                        or "N/A"
                    )
                    
                    if source_id not in citation_idx_map:
                        citation_idx_map[source_id] = len(citation_idx_map) + 1
                    
                    context_string += (
                        f'<source id="{citation_idx_map[source_id]}"'
                        + (f' name="{source_name}"' if source_name else "")
                        + f">{document_text}</source>\n"
                    )
        
        context_string = context_string.strip()
        
        return AgentResult(
            content=f"Retrieved relevant information for '{query}':\n\n{context_string}",
            sources=sources,
            metadata={
                "query": query,
                "sources_count": len(sources),
                "queries_used": queries,
                "citation_map": citation_idx_map
            }
        )
```

## Agent Interaction Patterns

### Web Search + RAG Workflow
With Option A (separate agents), the typical workflow becomes:

1. **LLM decides to search web**: Calls `search_web` agent
2. **Web Search Agent**:
   - Generates optimized search queries
   - Executes web search
   - Returns file references (not content)
   - Stores results in shared context
3. **LLM decides to query results**: Calls `query_documents` agent
4. **RAG Agent**:
   - Accesses web search files from context
   - Generates retrieval queries
   - Performs embedding/retrieval on web content
   - Returns formatted context with citations

### Agent Context Sharing
```python
class ChatContext(BaseModel):
    user: UserModel
    request: Request
    messages: List[Dict]
    metadata: Dict[str, Any]  # Shared state between agents
    app_state: Any
    event_emitter: Callable
    event_caller: Callable
    
    # Agent results are stored in metadata["files"] for cross-agent access
    def add_files(self, files: List[Dict]):
        if "files" not in self.metadata:
            self.metadata["files"] = []
        self.metadata["files"].extend(files)
```

### Agent Dependencies
While agents are independent, they can work together through shared context:
- **Web Search** → populates `metadata["files"]` with web sources
- **RAG** → processes any files in `metadata["files"]` (uploaded docs + web results)
- **Memory** → adds context to conversation history
- **Code/Image** → independent operations

## Migration Strategy

### Phase 1: Foundation (Week 1-2)
- [ ] Create agent base classes and interfaces
- [ ] Implement agent registry system
- [ ] Create chat context wrapper with shared state
- [ ] Add agent loading mechanism to `process_chat_payload()`
- [ ] Keep existing handlers as fallback

### Phase 2: Agent Implementation (Week 3-4)
- [ ] Implement Memory Agent (independent)
- [ ] Implement Web Search Agent (populates shared files)
- [ ] Implement RAG Agent (consumes shared files)
- [ ] Implement Image Generation Agent (independent)
- [ ] Implement Code Interpreter Agent (independent)
- [ ] Add agents to tools_dict in parallel with existing handlers

### Phase 3: Integration Testing (Week 5)
- [ ] Test individual agent functionality
- [ ] Test web search → RAG agent workflow
- [ ] Test agent combinations and shared context
- [ ] Add feature flag to switch between old/new systems
- [ ] Performance testing and optimization
- [ ] User acceptance testing

### Phase 4: Migration (Week 6)
- [ ] Default to agent-based system
- [ ] Remove old hardcoded feature handlers
- [ ] Clean up deprecated code
- [ ] Update documentation

### Phase 5: Enhancement (Week 7+)
- [ ] Add agent composition capabilities
- [ ] Implement explicit agent dependencies
- [ ] Add agent configuration UI
- [ ] Performance monitoring and analytics
- [ ] Smart agent suggestion based on context

## Implementation Details

### Agent Registry
```python
class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, ChatAgent] = {}
    
    def register(self, agent: ChatAgent):
        self._agents[agent.name] = agent
    
    def get_all_tools(self) -> Dict[str, Dict]:
        return {
            name: {
                "spec": agent.get_tool_spec(),
                "callable": agent.execute,
                "agent": agent
            }
            for name, agent in self._agents.items()
        }
    
    def get_agent(self, name: str) -> Optional[ChatAgent]:
        return self._agents.get(name)

# Global registry
agent_registry = AgentRegistry()
```

### Modified process_chat_payload()
```python
async def process_chat_payload(request, form_data, user, metadata, model):
    # ... existing setup code ...
    
    # Create chat context with shared state
    context = ChatContext(
        user=user,
        request=request,
        messages=form_data["messages"],
        metadata=metadata,  # This will be shared between agents
        app_state=request.app.state,
        event_emitter=event_emitter,
        event_caller=event_call
    )
    
    # Load agents into tools_dict with context binding
    agent_tools = {}
    for name, agent in agent_registry.get_all_agents().items():
        agent_tools[name] = {
            "spec": agent.get_tool_spec(),
            "callable": lambda params, ctx=context, ag=agent: ag.execute(params, ctx),
            "agent": agent
        }
    
    # Add agent tools to existing tools_dict
    tools_dict.update(agent_tools)
    
    # Remove old hardcoded feature handling
    # features = form_data.pop("features", None)
    # if features:
    #     # Old feature handlers removed
    
    # Let existing chat_completion_tools_handler manage everything
    if tools_dict:
        form_data, flags = await chat_completion_tools_handler(
            request, form_data, extra_params, user, models, tools_dict
        )
        sources.extend(flags.get("sources", []))
    
    # Handle any files added by agents (e.g., web search results)
    if context.metadata.get("files"):
        form_data["metadata"]["files"] = context.metadata["files"]
    
    # ... rest of existing code ...
```

## Testing Strategy

### Unit Tests
- [ ] Test each agent implementation independently
- [ ] Test agent registry functionality
- [ ] Test chat context creation and shared state
- [ ] Test tool specification generation
- [ ] Test agent context sharing mechanisms

### Integration Tests
- [ ] Test web search → RAG agent workflow
- [ ] Test agent execution within chat flow
- [ ] Test multiple agent combinations
- [ ] Test shared context between agents
- [ ] Test error handling and fallbacks
- [ ] Test performance under load

### End-to-End Tests
- [ ] Test complete chat flows with agents
- [ ] Test web search + document query scenarios
- [ ] Test UI integration with new agent system
- [ ] Test backward compatibility during migration
- [ ] Test feature parity with old system

### Agent Workflow Tests
- [ ] Test: Web search → Query documents workflow
- [ ] Test: Memory + RAG combination
- [ ] Test: Code execution with document context
- [ ] Test: Image generation with web research
- [ ] Test: Error handling when agents fail

## Success Metrics

### Functional
- [ ] All existing features work as agents
- [ ] LLM can dynamically choose which agents to use
- [ ] Agent combinations work correctly
- [ ] Performance is equivalent or better than current system

### Technical
- [ ] Code complexity reduced
- [ ] Feature coupling eliminated
- [ ] Extensibility improved
- [ ] Test coverage maintained or improved

### User Experience
- [ ] No regression in functionality
- [ ] Improved response relevance through dynamic agent selection
- [ ] Better composability of features
- [ ] Easier configuration and customization

## Risks and Mitigations

### Risk: Performance Degradation
**Mitigation**: Implement caching, optimize agent execution, performance testing

### Risk: LLM Tool Selection Quality
**Mitigation**: Improve tool descriptions, add examples, implement fallback logic

### Risk: Breaking Changes
**Mitigation**: Phased migration, feature flags, comprehensive testing

### Risk: Increased Complexity
**Mitigation**: Clear interfaces, good documentation, gradual rollout

## Conclusion

This refactor will transform Open WebUI from a rigid feature pipeline to a flexible, LLM-driven agent system. The migration strategy ensures minimal disruption while enabling powerful new capabilities for dynamic feature composition and extensibility.