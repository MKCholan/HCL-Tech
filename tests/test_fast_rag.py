"""Offline regression checks. No real credentials, Gemini calls, or index writes."""
import asyncio
from contextlib import aclosing
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hcl-tech-retrival'))
from embeddings.config import EMBEDDING_CONFIG
from fast_rag import FastRag, Gemini


class Collection:
    def __init__(self, count=1):
        self.queries = 0
        self.size = count

    def count(self):
        return self.size

    def query(self, **kwargs):
        self.queries += 1
        assert kwargs['n_results'] <= self.size
        return {'ids': [['chunk-1']], 'documents': [['A supporting document passage.']],
                'metadatas': [[{'source': 'guide.pdf', 'page': 2}]]}


def packet(text=None, finish=None, thought=False):
    candidate = {}
    if text is not None:
        candidate['content'] = {'parts': [{'text': text, 'thought': thought}]}
    if finish:
        candidate['finishReason'] = finish
    return ('data: ' + json.dumps({'candidates': [candidate]}) + '\n\n').encode()


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, gate=None):
        self.chunks, self.gate, self.closed = chunks, gate, False

    async def __aiter__(self):
        for i, chunk in enumerate(self.chunks):
            if i and self.gate is not None:
                await self.gate.wait()
            # Fragment SSE records across chunks, just like a real socket.
            midpoint = len(chunk) // 2
            yield chunk[:midpoint]
            yield chunk[midpoint:]

    async def aclose(self):
        self.closed = True


class RAGTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.status = 200
        self.gate = None
        self.finish = 'STOP'
        self.network_streams = []
        self.cfg = EMBEDDING_CONFIG

        async def handler(request):
            self.requests.append(request)
            body = json.loads(request.content)
            if request.url.path.endswith(':embedContent'):
                self.assertEqual(body['taskType'], 'RETRIEVAL_QUERY')
                self.assertEqual(body['outputDimensionality'], 768)
                return httpx.Response(200, json={'embedding': {'values': [3, 4] + [0] * 766}})
            if self.status != 200:
                return httpx.Response(self.status, json={'error': {'code': self.status}})
            self.assertEqual(request.url.params['alt'], 'sse')
            self.assertEqual(body['generationConfig']['thinkingConfig'], {'thinkingBudget': 0})
            stream = Stream([packet('internal thought', thought=True), packet('Answer [1].'), packet(finish=self.finish)], self.gate)
            self.network_streams.append(stream)
            return httpx.Response(200, headers={'Content-Type': 'text/event-stream'}, stream=stream)

        self.http = httpx.AsyncClient(base_url='https://generativelanguage.googleapis.com/v1beta/',
                                      transport=httpx.MockTransport(handler))
        self.rag = FastRag(Gemini(self.http, 'models/gemini-2.5-flash-lite', self.cfg), Collection())

    async def asyncTearDown(self):
        await self.http.aclose()

    async def collect(self, question='Explain FIR registration.', history=None):
        return [event async for event in self.rag.events(question, history or [])]

    async def test_one_generation_for_follow_up_and_valid_sources(self):
        events = await self.collect('Why is it required?', [{'role': 'user', 'content': 'Explain FIR registration.'}])
        self.assertEqual(len(self.requests), 2)  # embedding + generation, no rewrite
        self.assertIn('Explain FIR registration.', self.requests[0].content.decode())
        self.assertEqual(events[-1]['answer'], 'Answer [1].')
        self.assertEqual(events[-1]['sources'][0]['reference'], 1)
        self.assertEqual(events[-1]['sources'][0]['source'], 'guide.pdf')
        self.assertNotIn('internal thought', ''.join(e.get('text', '') for e in events))

    async def test_answer_cache_includes_history(self):
        await self.collect()
        cached = await self.collect()
        self.assertTrue(cached[-1]['cached'])
        self.assertEqual(len(self.requests), 2)
        different_context = await self.collect(history=[{'role': 'user', 'content': 'A different topic'}])
        self.assertFalse(different_context[-1]['cached'])
        self.assertEqual(len(self.requests), 3)  # reuses embedding, generates a new contextual answer

    async def test_stream_emits_text_before_generation_finishes(self):
        # Allow the first answer token, hold back only the finishing packet.
        gate = asyncio.Event()
        original = self.rag.gemini.stream

        async def held_stream(payload):
            yield {'text': 'First text'}
            await gate.wait()
            yield {'finish': 'STOP'}

        self.rag.gemini.stream = held_stream
        async with aclosing(self.rag.events('Question', [])) as iterator:
            event = await anext(iterator)
            while event['type'] != 'token':
                event = await anext(iterator)
            self.assertEqual(event['text'], 'First text')
            self.assertFalse(gate.is_set())
            gate.set()
            remaining = [event async for event in iterator]
            self.assertEqual(remaining[-1]['type'], 'done')
        self.rag.gemini.stream = original

    async def test_deadline_cancels_http_stream_and_allows_next_request(self):
        self.rag.total_seconds = 0.05
        self.gate = asyncio.Event()
        events = await asyncio.wait_for(self.collect(), timeout=1)
        self.assertEqual(events[-1]['status'], 504)
        self.assertIn('answer generation', events[-1]['error'])
        self.assertTrue(self.network_streams[-1].closed)
        self.assertEqual(len(self.requests), 2)
        self.gate = None
        self.rag.total_seconds = 1
        events = await self.collect()
        self.assertEqual(events[-1]['type'], 'done')
        self.assertFalse(events[-1]['cached'])

    async def test_overload_is_not_retried_or_cached(self):
        self.status = 503
        events = await self.collect()
        self.assertEqual(events[-1]['status'], 503)
        self.assertEqual(len(self.requests), 2)
        self.status = 200
        events = await self.collect()
        self.assertEqual(events[-1]['type'], 'done')
        self.assertEqual(len(self.requests), 3)

    async def test_quota_is_preserved(self):
        self.status = 429
        events = await self.collect()
        self.assertEqual(events[-1]['status'], 429)
        self.assertIn('quota', events[-1]['error'])
        self.assertEqual(len(self.requests), 2)

    async def test_truncated_output_is_labelled_and_not_cached(self):
        self.finish = 'MAX_TOKENS'
        events = await self.collect()
        self.assertTrue(events[-1]['truncated'])
        events = await self.collect()
        self.assertFalse(events[-1]['cached'])
        self.assertEqual(len(self.requests), 3)

    async def test_missing_finish_marker_is_not_success(self):
        self.finish = None
        events = await self.collect()
        self.assertEqual(events[-1]['status'], 502)
        self.assertNotIn('done', [event['type'] for event in events])

    async def test_empty_index_does_not_call_gemini(self):
        self.rag = FastRag(self.rag.gemini, Collection(count=0))
        events = await self.collect()
        self.assertEqual(events[-1]['sources'], [])
        self.assertEqual(len(self.requests), 0)

    async def test_browser_disconnect_closes_upstream_stream(self):
        iterator = self.rag.events('Question', [])
        async with aclosing(iterator):
            async for event in iterator:
                if event['type'] == 'token':
                    break
        self.assertTrue(self.network_streams[-1].closed)
        self.assertEqual(len(self.rag.answers.items), 0)

    async def test_fastapi_json_stream_and_health_routes(self):
        import api
        api.app.state.rag = self.rag
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
            health = (await client.get('/health')).json()
            self.assertEqual(health['build'], 'fast-rag-stream-v2')
            response = await client.post('/answer/stream', json={'question': 'Question'})
            records = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(records[-1]['type'], 'done')
            response = await client.post('/answer', json={'question': 'Question'})
            self.assertEqual(response.json()['answer'], 'Answer [1].')
            self.assertTrue(response.json()['cached'])
            self.assertEqual((await client.post('/answer', json={'question': '   '})).status_code, 400)
            self.status = 429
            response = await client.post('/answer', json={'question': 'A new question'})
            self.assertEqual(response.status_code, 429)


if __name__ == '__main__':
    unittest.main()
