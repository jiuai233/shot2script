'use strict';

function seconds(time) {
  const [m, s] = time.split(':').map(Number);
  return m * 60 + s;
}

// 仅使用标注 Schema 中的关键字；Schema 随页面嵌入，无需联网。
function checkSchema(value, schema, path = '标注') {
  const types = schema.type ? [].concat(schema.type) : [];
  const isType = type => type === 'null' ? value === null :
    type === 'array' ? Array.isArray(value) :
    type === 'object' ? value !== null && !Array.isArray(value) && typeof value === 'object' :
    type === 'integer' ? Number.isInteger(value) : typeof value === type;
  if (types.length && !types.some(isType)) throw Error(`${path}：类型不符`);
  if (schema.enum && !schema.enum.includes(value)) throw Error(`${path}：取值不符`);
  if (typeof value === 'string') {
    if (schema.pattern && !new RegExp(schema.pattern).test(value)) throw Error(`${path}：格式不符`);
    if (schema.minLength && Array.from(value).length < schema.minLength) throw Error(`${path}：不能为空`);
  }
  if (typeof value === 'number' && ((schema.minimum !== undefined && value < schema.minimum) ||
      (schema.maximum !== undefined && value > schema.maximum))) throw Error(`${path}：超出范围`);
  if (Array.isArray(value)) value.forEach((row, i) => checkSchema(row, schema.items, `${path}[${i}]`));
  if (value !== null && !Array.isArray(value) && typeof value === 'object') {
    for (const key of schema.required || []) if (!(key in value)) throw Error(`${path}：缺少 ${key}`);
    for (const key of Object.keys(value)) {
      if (schema.additionalProperties === false && !(key in schema.properties)) throw Error(`${path}：未知字段 ${key}`);
      if (schema.properties[key]) checkSchema(value[key], schema.properties[key], `${path}.${key}`);
    }
  }
}

function buildAnnotation(state, schema) {
  const result = JSON.parse(JSON.stringify(state));
  checkSchema(result, schema);
  const people = new Set(result.人物.map(row => row.id));
  for (const key of ['人物', '台词', '关键动作']) {
    const ids = new Set();
    for (const row of result[key]) {
      if (ids.has(row.id)) throw Error(`${key}：编号 ${row.id} 重复`);
      ids.add(row.id);
      if ('起' in row && seconds(row.止) < seconds(row.起)) throw Error(`${row.id}：结束早于开始`);
      for (const span of row.出现 || []) {
        const [start, end] = span.split('-');
        if (seconds(end) < seconds(start)) throw Error(`${row.id}：出现时间倒置`);
      }
      for (const field of ['说话人', '执行者', '对象']) {
        const id = row[field];
        if (id && /^H[0-9]+$/.test(id) && !people.has(id)) throw Error(`${row.id}：人物 ${id} 未登记`);
      }
      if (result.校对完成 && (row.待核对 ||
          (key === '台词' && (!row.说话人 || !row.形态)) ||
          (key === '关键动作' && !row.执行者))) throw Error(`${row.id}：仍有待核对内容`);
    }
  }
  result.台词.sort((a, b) => seconds(a.起) - seconds(b.起));
  result.关键动作.sort((a, b) => seconds(a.起) - seconds(b.起));
  return result;
}

if (typeof module !== 'undefined') module.exports = {buildAnnotation, checkSchema};

if (typeof document !== 'undefined') {
  const payload = JSON.parse(document.getElementById('sheet-data').textContent);
  let state = payload.annotation;
  const schema = payload.schema;
  const $ = id => document.getElementById(id);
  const status = message => { $('status').textContent = message; };
  function el(tag, text, parent) {
    const node = document.createElement(tag);
    if (text !== null) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  }
  function field(parent, row, key, options = null, convert = value => value) {
    const label = el('label', key, parent);
    const input = el(options ? 'select' : key === '文本' || key === '动作' || key === '外观' ? 'textarea' : 'input', null, label);
    if (options) for (const [value, name] of options) {
      const option = el('option', name, input);
      option.value = value;
    }
    input.value = Array.isArray(row[key]) ? row[key].join('，') : row[key] ?? '';
    const update = () => { row[key] = convert(input.value); status('有未导出的修改'); };
    input.addEventListener('input', update);
    input.addEventListener('change', () => {
      update();
      if (key === '名称' || key === 'id') render();
    });
    return input;
  }
  const nullable = value => value.trim() || null;
  function personOptions() {
    return [['', '待核对'], ...state.人物.map(row => [row.id, `${row.id} ${row.名称}`])];
  }
  function editor(parent, row, kind) {
    const box = el('fieldset', null, parent);
    box.dataset.kind = kind;
    box.dataset.id = row.id;
    el('legend', row.id, box);
    const grid = el('div', null, box);
    grid.className = 'fields';
    field(grid, row, 'id');
    if (kind === '人物') {
      field(grid, row, '名称'); field(grid, row, '外观');
      field(grid, row, '出现', null, value => value.split(/[，,\n]/).map(x => x.trim()).filter(Boolean));
    } else {
      field(grid, row, '起'); field(grid, row, '止');
      if (kind === '台词') {
        field(grid, row, '文本');
        field(grid, row, '说话人', personOptions(), nullable);
        field(grid, row, '形态', [['', '待核对'], ...['画内', '画外', '心声', '设备'].map(x => [x, x])], nullable);
      } else {
        field(grid, row, '执行者', personOptions(), nullable);
        for (const key of ['动作', '对象', '物件', '结果']) field(grid, row, key, null, key === '动作' ? value => value : nullable);
        field(grid, row, '类别', [[1, '1 物品交接'], [2, '2 身体接触'], [3, '3 位置变化'], [4, '4 关联对白'], [5, '5 可见结果']], Number);
      }
    }
    field(grid, row, '待核对');
    const remove = el('button', `删除 ${row.id}`, box);
    remove.type = 'button';
    remove.addEventListener('click', () => {
      if (!confirm(`删除 ${row.id}？`)) return;
      state[kind].splice(state[kind].indexOf(row), 1);
      render(); status(`已删除 ${row.id}；尚未导出`);
    });
  }
  function overlaps(a, b) {
    return seconds(a.起) < seconds(b.止) && seconds(a.止) >= seconds(b.起);
  }
  function render() {
    $('complete').checked = !!state.校对完成;
    for (const id of ['people', 'actions', 'shots', 'unplaced']) $(id).replaceChildren();
    for (const row of state.人物) editor($('people'), row, '人物');
    for (const row of state.关键动作) editor($('actions'), row, '关键动作');
    const placed = new Set();
    for (const shot of payload.shots) {
      const article = el('article', null, $('shots'));
      article.className = 'shot'; article.dataset.shot = shot.id;
      el('h3', `${shot.id}　${shot.起}—${shot.止}`, article);
      const seek = el('button', '定位到这个镜头', article);
      seek.type = 'button';
      seek.addEventListener('click', () => { $('video').currentTime = seconds(shot.起); $('video').focus(); });
      const strip = el('div', null, article); strip.className = 'frames';
      for (const frame of shot.frames) {
        const figure = el('figure', null, strip);
        const link = el('a', null, figure); link.href = frame.src;
        const image = el('img', null, link);
        image.src = frame.src; image.alt = `${shot.id} ${frame.时间} 参考帧`; image.loading = 'lazy';
        el('figcaption', frame.时间, figure);
      }
      const local = state.台词.filter(row => overlaps(row, shot));
      for (const row of local) {
        if (!placed.has(row.id)) { editor(article, row, '台词'); placed.add(row.id); }
        else el('p', `${row.id} 跨镜头延续：${row.文本}（在首次出现镜头编辑）`, article);
      }
    }
    for (const row of state.台词) if (!placed.has(row.id)) editor($('unplaced'), row, '台词');
    $('counts').textContent = `${payload.shots.length} 个镜头；${state.台词.length} 条台词；${state.人物.length} 个人物；${state.关键动作.length} 个关键动作`;
  }
  function nextId(key, prefix, width) {
    return prefix + String(Math.max(0, ...state[key].map(row => Number(row.id.slice(1)) || 0)) + 1).padStart(width, '0');
  }
  $('add-person').onclick = () => {
    state.人物.push({id: nextId('人物', 'H', 2), 名称: '', 外观: '', 出现: [], 待核对: ''});
    render(); $('people').lastElementChild.querySelector('input').focus();
  };
  $('add-action').onclick = () => {
    state.关键动作.push({id: nextId('关键动作', 'M', 3), 起: '00:00.0', 止: '00:00.1', 执行者: null, 动作: '', 对象: null, 物件: null, 结果: null, 类别: 1, 待核对: ''});
    render(); $('actions').lastElementChild.querySelector('input').focus();
  };
  $('add-line').onclick = () => {
    state.台词.push({id: nextId('台词', 'L', 3), 起: '00:00.0', 止: '00:00.1', 文本: '', 说话人: null, 形态: null, 待核对: ''});
    render(); $('shots').querySelector('fieldset:last-of-type input')?.focus();
  };
  $('complete').onchange = event => { state.校对完成 = event.target.checked; status('有未导出的修改'); };
  $('refresh').onclick = render;
  $('import').onchange = async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const loaded = buildAnnotation(JSON.parse(await file.text()), schema);
      if (loaded.集 !== state.集 || loaded.视频 !== state.视频) throw Error('导入文件与当前视频不一致');
      if (!confirm('用导入文件替换页面中尚未导出的修改？')) return;
      state = loaded; render(); status('已载入标注');
    } catch (error) { status(error.message); }
  };
  $('export').onclick = () => {
    try {
      const data = buildAnnotation(state, schema);
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2) + '\n'], {type: 'application/json'}));
      const link = el('a', null, document.body);
      link.href = url; link.download = '标注.json'; link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      status('标注.json 已导出');
    } catch (error) { status(error.message); $('status').focus(); }
  };
  window.addEventListener('beforeunload', event => { event.preventDefault(); event.returnValue = ''; });
  render();
}
