const assert = require('node:assert/strict');
const api = require(process.argv[2]);
const snapshot = [{id:'E01-S001',文本:'甲',起:'00:00.0',止:'00:01.0'}];
const items = [{句号:'E01-S001',主路:'甲',副路:'乙',字幕:'丙'}];
let content = null;
const storage = {getItem: () => content, setItem: (_key, value) => {content = value;}};
const bad = {getItem() {throw Error('禁止读取');},setItem() {throw Error('禁止写入');}};
const decisions = {'E01-S001':{句号:'E01-S001',决定:'副路',最终文本:'乙',听辨:'重叠',备注:'回听'}};
assert.equal(api.persist(storage,'key',snapshot,decisions),true);
assert.deepEqual(api.restore(storage,'key',snapshot,items).decisions,decisions);
assert.deepEqual(api.exportResults(snapshot,items,decisions),{声音快照:snapshot,声音摘要:snapshot,字幕参考:{'E01-S001':'丙'},结果:[decisions['E01-S001']]});
assert.equal(api.persist(bad,'key',snapshot,decisions),false);
assert.deepEqual(api.restore(bad,'key',snapshot,items),{decisions:{},available:false});
assert.deepEqual(api.restore(null,'key',snapshot,items),{decisions:{},available:false});
assert.deepEqual(api.exportResults(snapshot,items,decisions).结果,[decisions['E01-S001']]);
assert.deepEqual(api.restore(storage,'key',[{文本:'已变'}],items).decisions,{});
content = '{bad json';
assert.equal(api.restore(storage,'key',snapshot,items).available,false);
for (const source of ['主路','副路','字幕','手改','删除']) {
  const value = {句号:'E01-S001',决定:source,最终文本: source==='删除'?'':source==='手改'?'手写':items[0][source],听辨:'清楚',备注:''};
  assert.equal(api.exportResults(snapshot,items,{'E01-S001':value}).结果.length,1);
}
assert.equal(api.exportResults(snapshot,items,{'E01-S001':{...decisions['E01-S001'],最终文本:'别的'}}).结果.length,0);
assert.equal(api.exportResults(snapshot,items,{'E01-S001':{...decisions['E01-S001'],决定:'手改',最终文本:''}}).结果.length,0);
console.log('导出、恢复、存储失效与所有决定来源：通过');
const draft={'E01-S001':{句号:'E01-S001',决定:'',最终文本:'甲',听辨:'听不清',备注:'先记下疑问'}};
api.persist(storage,'key',snapshot,draft);
assert.deepEqual(api.restore(storage,'key',snapshot,items).decisions,draft);
assert.equal(api.exportResults(snapshot,items,draft).结果.length,0);
