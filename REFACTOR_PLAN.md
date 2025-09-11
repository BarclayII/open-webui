# Open WebUI Chat Processing Refactor Plan

## Executive Summary

This document outlines the refactor plan to transform Open WebUI's chat processing from a hardcoded feature pipeline to a dynamic, LLM-driven tool system where features (memory, web search, image generation, code interpreter, RAG) are converted to invocable tools that leverage the existing iterative tool call infrastructure.

**Key Discovery**: Open WebUI already has sophisticated iterative tool calling in [`process_chat_response()`](backend/open_webui/utils/middleware.py:2212-2386). We just need to convert features to tools rather than build new infrastructure.

## Current Architecture Analysis

### Current Flow
```
Chat Request → process_chat_payload() → Fixed Feature Pipeline → chat_completion_handler() → process_chat_response()
                                                                                                    ↳ Already handles iterative tool calls!
```

### Existing Iterative Tool Call Infrastructure ✅

**Key Discovery**: [`process_chat_response()`](backend/open_webui/utils/middleware.py:2212-2386) already implements sophisticated iterative tool calling:

- ✅ **Multiple tool call rounds**: Loop until no more tools needed
- ✅ **LLM continuation**: Feeds tool results back to LLM for next iteration  
- ✅ **Proper message history**: Builds OpenAI-compatible tool call messages
- ✅ **Streaming preservation**: Maintains streaming during tool execution
- ✅ **Retry limits**: [`CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES`](backend/open_webui/utils/middleware.py:101) prevents infinite loops
- ✅ **Error handling**: Graceful tool failure handling

### Current Feature Pipeline (Fixed Order in process_chat_payload)
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
- **Iterative Tool Loop**: [`middleware.py:2212-2386`](backend/open_webui/utils/middleware.py:2212-2386) - Existing tool call system

### Current Feature Handlers (To Be Converted)
| Feature | Handler Function | Location | Current Behavior | Target Tool |
|---------|------------------|----------|------------------|-------------|
| Memory | [`chat_memory_handler()`](backend/open_webui/utils/middleware.py:324) | Preprocessing | Modifies `form_data["messages"]` | `query_memory` tool |
| Web Search | [`chat_web_search_handler()`](backend/open_webui/utils/middleware.py:363) | Preprocessing | Adds to `form_data["files"]` | `search_web` tool |
| Image Generation | [`chat_image_generation_handler()`](backend/open_webui/utils/middleware.py:525) | Preprocessing | Modifies `form_data["messages"]` | `generate_image` tool |
| Code Interpreter | Inline setup | [`middleware.py:907`](backend/open_webui/utils/middleware.py:907) | Adds system prompt | `execute_code` tool |
| RAG/Files | [`chat_completion_files_handler()`](backend/open_webui/utils/middleware.py:626) | Preprocessing | Returns sources for context | `query_documents` tool |

### Current Issues
1. **Fixed Pipeline Order**: Features execute in predetermined sequence in preprocessing
2. **Mixed Interfaces**: Inconsistent input/output patterns across handlers
3. **No Dynamic Discovery**: LLM cannot choose which features to use when
4. **Preprocessing Lock-in**: Features modify form_data before LLM sees the request
5. **Limited Iterative Capability**: Cannot chain feature usage based on results

## Target Architecture

### New Simplified Flow
```
Chat Request → process_chat_payload() → Load Feature Tools → chat_completion_handler() → process_chat_response() 
                     ↳ Convert features to tools                                              ↳ Existing iterative tool loop
```

### Design Principles
1. **Leverage Existing Infrastructure**: Use proven tool call system in [`process_chat_response()`](backend/open_webui/utils/middleware.py:2212-2386)
2. **Simple Tool Conversion**: Convert feature handlers to OpenAI-compatible tools
3. **Dynamic LLM Selection**: LLM decides which tools to use and when
4. **Minimal Changes**: Reuse existing streaming, events, and tool infrastructure
5. **Backward Compatible**: Maintain all current functionality

### Core Changes Required

#### 1. Remove Fixed Feature Pipeline
**In [`process_chat_payload()`](backend/open_webui/utils/middleware.py:890-915):**
```python
# REMOVE: Fixed feature execution
features = form_data.pop("features", None)
if features:
    if "memory" in features and features["memory"]:
        form_data = await chat_memory_handler(...)  # REMOVE
    if "web_search" in features and features["web_search"]:
        form_data = await chat_web_search_handler(...)  # REMOVE
    # ... etc
```

#### 2. Add Feature Tools to tools_dict
**In [`process_chat_payload()`](backend/open_webui/utils/middleware.py:965-981):**
```python
# NEW: Convert enabled features to tools
feature_tools = {}
if features:
    if features.get("memory"):
        feature_tools["query_memory"] = create_memory_tool(request, user, extra_params)
    if features.get("web_search"):
        feature_tools["search_web"] = create_web_search_tool(request, user, extra_params)
    if features.get("image_generation"):
        feature_tools["generate_image"] = create_image_tool(request, user, extra_params)
    if features.get("code_interpreter"):
        feature_tools["execute_code"] = create_code_tool(request, user, extra_params)

# Add feature tools to existing tools_dict
tools_dict.update(feature_tools)
```

#### 3. Simple Tool Creation Functions
```python
def create_memory_tool(request, user, extra_params):
    async def execute_memory_query(**params):
        # Reuse existing chat_memory_handler logic
        query = params.get("query", "")
        k = params.get("k", 3)
        
        results = await query_memory(
            request, 
            QueryMemoryForm(content=query, k=k), 
            user
        )
        
        user_context = ""
        if results and hasattr(results, "documents"):
            for doc_idx, doc in enumerate(results.documents[0]):
                created_at = "Unknown Date"
                if results.metadatas[0][doc_idx].get("created_at"):
                    timestamp = results.metadatas[0][doc_idx]["created_at"]
                    created_at = time.strftime("%Y-%m-%d", time.localtime(timestamp))
                user_context += f"{doc_idx + 1}. [{created_at}] {doc}\n"
        
        return f"Retrieved memory context:\n{user_context}"
    
    return {
        "spec": {
            "type": "function",
            "function": {
                "name": "query_memory",
                "description": "Query user's memory for relevant past conversations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Memory search query"},
                        "k": {"type": "integer", "description": "Number of results", "default": 3}
                    },
                    "required": ["query"]
                }
            }
        },
        "callable": execute_memory_query
    }
```

## Feature Tool Implementations

### 1. Memory Tool
**Convert [`chat_memory_handler()`](backend/open_webui/utils/middleware.py:324)**
```python
def create_memory_tool(request, user, extra_params):
    async def execute_memory_query(**params):
        query = params.get("query", "")
        k = params.get("k", 3)
        
        # Reuse existing memory query logic
        results = await query_memory(request, QueryMemoryForm(content=query, k=k), user)
        
        # Format results same as current handler
        user_context = ""
        if results and hasattr(results, "documents"):
            for doc_idx, doc in enumerate(results.documents[0]):
                created_at = "Unknown Date"
                if results.metadatas[0][doc_idx].get("created_at"):
                    timestamp = results.metadatas[0][doc_idx]["created_at"]
                    created_at = time.strftime("%Y-%m-%d", time.localtime(timestamp))
                user_context += f"{doc_idx + 1}. [{created_at}] {doc}\n"
        
        return f"Retrieved memory context:\n{user_context}"
    
    return {
        "spec": {
            "type": "function",
            "function": {
                "name": "query_memory",
                "description": "Query user's memory for relevant past conversations and context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query to search in user's memory"},
                        "k": {"type": "integer", "description": "Number of memory items to retrieve", "default": 3}
                    },
                    "required": ["query"]
                }
            }
        },
        "callable": execute_memory_query
    }
```

### 2. Web Search Tool
**Convert [`chat_web_search_handler()`](backend/open_webui/utils/middleware.py:363)**
```python
def create_web_search_tool(request, user, extra_params):
    async def execute_web_search(**params):
        query = params.get("query", "")
        event_emitter = extra_params["__event_emitter__"]
        
        # Emit status events (same as current handler)
        await event_emitter({
            "type": "status",
            "data": {"action": "web_search", "description": "Searching the web", "done": False},
        })
        
        # Generate search queries (reuse existing logic)
        try:
            res = await generate_queries(request, {
                "model": extra_params["__metadata__"]["model"],
                "messages": extra_params["__metadata__"]["messages"],
                "prompt": query,
                "type": "web_search",
            }, user)
            
            response = res["choices"][0]["message"]["content"]
            # Parse queries (same logic as current handler)
            queries = [query]  # Simplified for example
            
        except Exception as e:
            queries = [query]
        
        # Execute web search (reuse existing logic)
        results = await process_web_search(request, SearchForm(queries=queries), user=user)
        
        # Store files in shared metadata for RAG tool to access later
        files = []
        if results:
            if results.get("collection_names"):
                for collection_name in results.get("collection_names"):
                    files.append({
                        "collection_name": collection_name,
                        "name": ", ".join(queries),
                        "type": "web_search",
                        "urls": results["filenames"],
                        "queries": queries,
                    })
        
        # Add files to metadata for other tools to access
        if "files" not in extra_params["__metadata__"]:
            extra_params["__metadata__"]["files"] = []
        extra_params["__metadata__"]["files"].extend(files)
        
        await event_emitter({
            "type": "status",
            "data": {"action": "web_search", "description": f"Found {len(results.get('filenames', []))} sources", "done": True},
        })
        
        return f"Web search completed for '{query}'. Found {len(results.get('filenames', []))} sources. Use query_documents to retrieve specific information from these sources."
    
    return {
        "spec": {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query or topic to search for"}
                    },
                    "required": ["query"]
                }
            }
        },
        "callable": execute_web_search
    }
```

### 3. RAG/Documents Tool
**Convert [`chat_completion_files_handler()`](backend/open_webui/utils/middleware.py:626)**
```python
def create_rag_tool(request, user, extra_params):
    async def execute_document_query(**params):
        query = params.get("query", "")
        k = params.get("k", 5)
        
        # Get files from shared metadata (includes uploaded files, web search results)
        files = extra_params["__metadata__"].get("files", [])
        if not files:
            return "No documents or sources available to query. Try uploading documents or using web search first."
        
        # Generate retrieval queries (reuse existing logic)
        try:
            queries_response = await generate_queries(request, {
                "model": extra_params["__metadata__"]["model"],
                "messages": extra_params["__metadata__"]["messages"],
                "type": "retrieval",
            }, user)
            
            # Parse queries (same logic as current handler)
            queries = [query]  # Simplified for example
        except:
            queries = [query]
        
        # Use existing RAG logic
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                sources = await loop.run_in_executor(
                    executor,
                    lambda: get_sources_from_items(
                        request=request,
                        items=files,
                        queries=queries,
                        embedding_function=lambda q, prefix: request.app.state.EMBEDDING_FUNCTION(q, prefix=prefix, user=user),
                        k=k,
                        reranking_function=(
                            (lambda sentences: request.app.state.RERANKING_FUNCTION(sentences, user=user))
                            if request.app.state.RERANKING_FUNCTION else None
                        ),
                        k_reranker=request.app.state.config.TOP_K_RERANKER,
                        r=request.app.state.config.RELEVANCE_THRESHOLD,
                        hybrid_bm25_weight=request.app.state.config.HYBRID_BM25_WEIGHT,
                        hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                        full_context=request.app.state.config.RAG_FULL_CONTEXT,
                        user=user,
                    ),
                )
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
        
        if not sources:
            return f"No relevant information found for query: '{query}'"
        
        # Format retrieved context (same as current handler)
        context_string = ""
        citation_idx_map = {}
        
        for source in sources:
            if "document" in source:
                for document_text, document_metadata in zip(source["document"], source["metadata"]):
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
        
        return f"Retrieved relevant information for '{query}':\n\n{context_string.strip()}"
    
    return {
        "spec": {
            "type": "function",
            "function": {
                "name": "query_documents",
                "description": "Query uploaded documents, knowledge base, and web search results for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query to search in documents and knowledge sources"},
                        "k": {"type": "integer", "description": "Number of relevant chunks to retrieve", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        },
        "callable": execute_document_query
    }
```

### 4. Image Generation Tool
**Convert [`chat_image_generation_handler()`](backend/open_webui/utils/middleware.py:525)**
```python
def create_image_tool(request, user, extra_params):
    async def execute_image_generation(**params):
        prompt = params.get("prompt", "")
        event_emitter = extra_params["__event_emitter__"]
        
        await event_emitter({
            "type": "status",
            "data": {"description": "Generating an image", "done": False},
        })
        
        try:
            # Reuse existing image generation logic
            images = await image_generations(
                request=request,
                form_data=GenerateImageForm(prompt=prompt),
                user=user,
            )
            
            await event_emitter({
                "type": "status",
                "data": {"description": "Generated an image", "done": True},
            })
            
            await event_emitter({
                "type": "files",
                "data": {"files": [{"type": "image", "url": image["url"]} for image in images]},
            })
            
            return "Image has been generated successfully"
            
        except Exception as e:
            await event_emitter({
                "type": "status",
                "data": {"description": "An error occurred while generating an image", "done": True},
            })
            
            return "Unable to generate an image, an error occurred"
    
    return {
        "spec": {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on a text prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Text prompt describing the image to generate"}
                    },
                    "required": ["prompt"]
                }
            }
        },
        "callable": execute_image_generation
    }
```

### 5. Code Execution Tool
**Convert Code Interpreter Setup**
```python
def create_code_tool(request, user, extra_params):
    async def execute_code(**params):
        code = params.get("code", "")
        session_id = extra_params["__metadata__"].get("session_id")
        
        if not session_id:
            return "Code execution requires a valid session"
        
        try:
            # Apply security restrictions (same as current implementation)
            if CODE_INTERPRETER_BLOCKED_MODULES:
                blocking_code = textwrap.dedent(f"""
                    import builtins
                    BLOCKED_MODULES = {CODE_INTERPRETER_BLOCKED_MODULES}
                    
                    _real_import = builtins.__import__
                    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
                        if name.split('.')[0] in BLOCKED_MODULES:
                            importer_name = globals.get('__name__') if globals else None
                            if importer_name == '__main__':
                                raise ImportError(f"Direct import of module {{name}} is restricted.")
                        return _real_import(name, globals, locals, fromlist, level)
                    
                    builtins.__import__ = restricted_import
                """)
                code = blocking_code + "\n" + code
            
            # Execute code using existing logic
            if request.app.state.config.CODE_INTERPRETER_ENGINE == "pyodide":
                output = await extra_params["__event_call__"]({
                    "type": "execute:python",
                    "data": {"id": str(uuid4()), "code": code, "session_id": session_id},
                })
            elif request.app.state.config.CODE_INTERPRETER_ENGINE == "jupyter":
                output = await execute_code_jupyter(
                    request.app.state.config.CODE_INTERPRETER_JUPYTER_URL,
                    code,
                    request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_TOKEN,
                    request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD,
                    request.app.state.config.CODE_INTERPRETER_JUPYTER_TIMEOUT,
                )
            else:
                output = {"stdout": "Code interpreter engine not configured."}
            
            # Format output (same as current implementation)
            result_content = ""
            if isinstance(output, dict):
                stdout = output.get("stdout", "")
                stderr = output.get("stderr", "")
                result = output.get("result", "")
                
                if stdout:
                    result_content += f"**Output:**\n```\n{stdout}\n```\n\n"
                if stderr:
                    result_content += f"**Errors:**\n```\n{stderr}\n```\n\n"
                if result:
                    result_content += f"**Result:**\n```\n{result}\n```\n\n"
            
            return f"```python\n{code}\n```\n\n{result_content.strip() or 'Code executed successfully (no output)'}"
            
        except Exception as e:
            return f"Code execution failed:\n```python\n{code}\n```\n\nError: {str(e)}"
    
    return {
        "spec": {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute Python code with persistent session state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 30}
                    },
                    "required": ["code"]
                }
            }
        },
        "callable": execute_code
    }
```

## Migration Strategy (Simplified)

### Phase 1: Tool Conversion (Week 1-2)
- [ ] Create feature tool conversion functions
- [ ] Test each tool individually  
- [ ] Ensure backward compatibility
- [ ] Add feature flag for old vs new system

### Phase 2: Integration (Week 3)
- [ ] Modify [`process_chat_payload()`](backend/open_webui/utils/middleware.py:890-915) to remove fixed pipeline
- [ ] Add feature tools to existing `tools_dict`
- [ ] Test with existing iterative tool call system
- [ ] Performance testing

### Phase 3: Migration (Week 4) 
- [ ] Enable new system by default
- [ ] Remove old feature handlers
- [ ] Clean up deprecated code
- [ ] Update documentation

## Key Benefits

### 1. **Leverages Existing Infrastructure**
- Uses proven tool call system in [`process_chat_response()`](backend/open_webui/utils/middleware.py:2212-2386)
- No new streaming or event systems needed
- Minimal changes to core architecture

### 2. **Dynamic Feature Selection**
- LLM decides which tools to use and when
- Can chain tools based on results (e.g., web search → RAG)
- Adaptive strategy based on context

### 3. **Complex Reasoning Capability**
- Multiple tool call iterations
- Tool results inform subsequent LLM reasoning
- Natural multi-step workflows

### 4. **Simple Implementation** 
- Convert handlers to tool callables
- Add to existing `tools_dict`
- Reuse all existing logic

## Example Workflow

```
User: "Search for Python tutorials and summarize the first result"

1. LLM Response: "I'll search for Python tutorials first"
   Tool Call: search_web(query="Python tutorials")

2. Tool Execution: Returns web search results stored in metadata

3. LLM Continuation: "Now let me get the content from the first result"
   Tool Call: query_documents(query="Python tutorial content first result")

4. Tool Execution: RAG retrieves content from web search results

5. LLM Final: "Here's a summary of the Python tutorial: ..."
   No more tool calls needed
```

## Risks and Mitigations

### Risk: Performance Impact
**Mitigation**: Existing tool system is already optimized; feature tools reuse existing logic

### Risk: LLM Tool Selection Quality  
**Mitigation**: Improve tool descriptions; existing system already works for external tools

### Risk: Breaking Changes
**Mitigation**: Feature flag for gradual migration; maintain backward compatibility

## Conclusion

This simplified refactor leverages Open WebUI's existing iterative tool call infrastructure to enable complex reasoning with minimal architectural changes. By converting features to tools rather than building new agent systems, we achieve the goal of dynamic, LLM-driven feature selection while maintaining stability and reusing proven components.