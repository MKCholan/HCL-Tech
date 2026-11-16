const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { readAnswerStream } = require('../public/stream.js');
const originalFetch = global.fetch;
process.env.RAG_TIMEOUT_MS = '100';
const app = require('../server.cjs');
const server = app.listen(0);
after(() => { global.fetch = originalFetch; server.close(); });
function request(body, pathname='/api/chat') {
  return new Promise((resolve, reject) => {
    const req = http.request({ port: server.address().port, path: pathname, method: 'POST', headers: {'Content-Type':'application/json'} }, res => {
      let data=''; res.on('data', b => data+=b); res.on('end', () => resolve({ status:res.statusCode, data }));
    }); req.on('error', reject); req.end(JSON.stringify(body));
  });
}
function streamResponse(records) {
  return new Response(records.map(record=>JSON.stringify(record)+'\n').join(''), {headers:{'Content-Type':'application/x-ndjson'}});
}
test('consecutive JSON requests work and history is bounded', async () => {
  let calls=0;
  global.fetch=async(url,options)=>{
    calls++;
    assert.ok(options.signal);
    const body=JSON.parse(options.body);
    assert.equal(body.history.length,4);
    assert.ok(body.history.every(m=>m.content.length<=1500));
    return Response.json({answer:'Answer '+calls,sources:[]});
  };
  for(let i=1;i<=3;i++) {
    const r=await request({question:'Question',history:Array(20).fill({role:'user',content:'x'.repeat(2000)})});
    assert.equal(r.status,200);assert.equal(JSON.parse(r.data).answer,'Answer '+i);
  }
});
test('quota and service failures retain their useful messages',async()=>{
  global.fetch=async()=>Response.json({detail:'Gemini quota reached'},{status:429});
  const r=await request({question:'Question'});
  assert.equal(r.status,429);assert.match(JSON.parse(r.data).error,/quota/);
  global.fetch=async()=>{throw new TypeError('fetch failed')};
  const unavailable=await request({question:'Question'});
  assert.equal(unavailable.status,503);assert.match(JSON.parse(unavailable.data).error,/port 8000/);
});
test('stream reaches browser parser before the upstream finishes',async()=>{
  let release;
  const gate=new Promise(resolve=>{release=resolve});
  global.fetch=async()=>new Response(new ReadableStream({async start(controller){
    controller.enqueue(new TextEncoder().encode(JSON.stringify({type:'token',text:'First text'})+'\n'));
    await gate;
    controller.enqueue(new TextEncoder().encode(JSON.stringify({type:'done',answer:'First text'})+'\n'));
    controller.close();
  }}),{headers:{'Content-Type':'application/x-ndjson'}});
  const url=`http://127.0.0.1:${server.address().port}/api/chat/stream`;
  const response=await originalFetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:'Question'})});
  let tokens=0;
  const events=[];
  await readAnswerStream(response.body,event=>{
    events.push(event);
    if(event.type==='token'){tokens++;release();}
  });
  assert.equal(tokens,1);assert.equal(events.at(-1).type,'done');
});
test('deadline applies to a stalled response body and the next request works',async()=>{
  let aborted=false;
  global.fetch=async(url,options)=>new Response(new ReadableStream({start(controller){
    options.signal.addEventListener('abort',()=>{aborted=true;controller.error(options.signal.reason)});
    controller.enqueue(new TextEncoder().encode('{"type":"status","message":"Waiting"}\n'));
  }}),{headers:{'Content-Type':'application/x-ndjson'}});
  const r=await request({question:'Question'},'/api/chat/stream');
  const records=r.data.trim().split('\n').map(line=>JSON.parse(line));
  assert.equal(records.at(-1).status,504);assert.equal(aborted,true);
  global.fetch=async()=>Response.json({answer:'Recovered'});
  assert.equal((await request({question:'Next question'})).status,200);
});
test('malformed history and old backend responses are rejected',async()=>{
  assert.equal((await request({question:'Question',history:[null]})).status,400);
  global.fetch=async()=>Response.json({answer:'Old response'});
  const r=await request({question:'Question'},'/api/chat/stream');
  assert.equal(r.status,502);assert.match(JSON.parse(r.data).error,/older version/);
});
test('NDJSON parser handles split Unicode and records without trailing newline',async()=>{
  const bytes=new TextEncoder().encode('{"type":"token","text":"✓ தமிழ்"}\n{"type":"done"}');
  const events=[];
  const stream=new ReadableStream({start(c){for(const byte of bytes)c.enqueue(Uint8Array.of(byte));c.close()}});
  await readAnswerStream(stream,event=>events.push(event));
  assert.equal(events[0].text,'✓ தமிழ்');assert.equal(events[1].type,'done');
});
