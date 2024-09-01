import os, asyncio
async def progress_generator(request, case_id: str, file_keys: List[str]):
    async def send_event(event, data):
        if await request.is_disconnected():
            raise asyncio.CancelledError()
        yield {"event": event, "data": data}
        # Introducing a small delay to allow for event processing
        await asyncio.sleep(0.01)

    try:
        documents = []
        # Fetch documents with progress updates
        async for progress_update in fetch_all(file_keys):
            if isinstance(progress_update, str):
                async for event in send_event("update", progress_update):
                    yield event
            else:
                documents = progress_update  # This is the final list of documents

        # Create vectorstore
        total_docs = len(documents)
        processing_batch_size = 200
        update_frequency = 10
        for i in range(0, total_docs, processing_batch_size):
            batch = documents[i:i+processing_batch_size]
            create_vectorstore(batch, case_id)
            
            # Update progress more frequently than the actual processing batch
            for j in range(i, min(i + processing_batch_size, total_docs), update_frequency):
                progress = min(j / total_docs * 100, 100)
                async for event in send_event("update", f"Embedding progress: {progress:.2f}%"):
                    yield event

        async for event in send_event("update", "Vectorstore creation completed"):
            yield event

    except asyncio.CancelledError:
        async for event in send_event("update", "Client disconnected, stopping process"):
            yield event
    except Exception as e:
        async for event in send_event("error", str(e)):
            yield event

