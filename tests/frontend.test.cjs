const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { readAnswerStream } = require('../public/stream.js');
class Element {
  constructor() { this.children=[];this.textContent='';this.value='';this.disabled=false; }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); }
  remove() { this.removed=true; }
  focus() {}
  addEventListener() {}
}
function setup(fetch) {
  const nodes=Object.fromEntries(['question-form','question','submit-button','chat','status','welcome'].map(id=>[id,new Element()]));
  const context=vm.createContext({document:{getElementById:id=>nodes[id],createElement:()=>new Element()},
    fetch,readAnswerStream,AbortController,DOMException,TypeError,performance,setTimeout,clearTimeout,setInterval,clearInterval});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../public/app.js'),'utf8'),context);
  return {nodes,context,submit:()=>vm.runInContext('submitQuestion({preventDefault(){}})',context)};
}
const jsonLines=records=>new Response(records.map(e=>JSON.stringify(e)+'\n').join(''));
function record(type,extras={}) { return {type,...extras}; }
test('frontend displays text before done and retains only completed conversation turns',async()=>{
  let release;
  const gate=new Promise(r=>{release=r});
  const {nodes,context,submit}=setup(async()=>new Response(new ReadableStream({async start(c){
    c.enqueue(new TextEncoder().encode(JSON.stringify(record('token',{text:'First text'}))+'\n'));
    await gate;
    c.enqueue(new TextEncoder().encode(JSON.stringify(record('done',{answer:'First text [1].',sources:[{reference:1,source:'guide.pdf'}]}))+'\n'));
    c.close();
  }})));
  nodes.question.value='Question';
  const pending=submit();
  await new Promise(r=>setImmediate(r));
  assert.equal(nodes['submit-button'].textContent,'Stop');
  assert.equal(nodes.chat.children.at(-1).children[1].textContent,'First text');
  assert.equal(vm.runInContext('history.length',context),0);
  release();await pending;
  assert.equal(nodes['submit-button'].textContent,'Submit');
  assert.equal(vm.runInContext('history.length',context),2);
  assert.match(nodes.chat.children.at(-1).children[2].textContent,/\[1\] guide.pdf/);
});
test('frontend handles streamed overload and can answer a subsequent question',async()=>{
  let fail=true;
  const {nodes,context,submit}=setup(async()=> fail
    ?jsonLines([record('error',{status:503,error:'Gemini is overloaded'})])
    :jsonLines([record('token',{text:'Answer'}),record('done',{answer:'Answer',sources:[]})]));
  nodes.question.value='Question';await submit();
  assert.equal(nodes.question.value,'Question');
  assert.equal(nodes['submit-button'].disabled,false);
  assert.equal(vm.runInContext('history.length',context),0);
  assert.match(nodes.chat.children.at(-1).textContent,/overloaded/);
  fail=false;await submit();
  assert.equal(vm.runInContext('history.length',context),2);
  assert.equal(nodes['submit-button'].textContent,'Submit');
});
test('Stop cancels an unfinished stream and excludes it from history',async()=>{
  const {nodes,context,submit}=setup(async(url,options)=>new Response(new ReadableStream({start(c){
    options.signal.addEventListener('abort',()=>c.error(options.signal.reason));
    c.enqueue(new TextEncoder().encode(JSON.stringify(record('token',{text:'Partial text'}))+'\n'));
  }})));
  nodes.question.value='Question';const pending=submit();
  await new Promise(r=>setImmediate(r));
  await submit();await pending;
  assert.equal(vm.runInContext('history.length',context),0);
  assert.equal(nodes['submit-button'].textContent,'Submit');
  assert.match(nodes.chat.children.at(-1).textContent,/stopped/);
  assert.match(nodes.chat.children.at(-2).children[0].textContent,/incomplete/);
});
