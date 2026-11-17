/* Tapes: long recordings (rows = polls, columns = the channels of the line). Pool -> generate -> edit (add, delete, move) -> CSV + labels.
   The tape description lives here; every pixel and box comes from the API. */
'use strict';
const tapeCell=12,tapeMapHeight=720;
const tapeUI={pool:[],poolSelected:new Set(),addIndex:-1,addPrepared:null,addToken:0,
  tape:null,preview:null,levels:null,mapBase:null,selected:-1,placing:false,drag:null,undo:[],redo:[],
  top:0,rows:60,busy:false,message:'',output:'',directory:'synthetic_data',tapeSplit:null,name:'',values:'levels',header:false,
  rules:{length:400,objects:20,layout:'random',jitter:0.3,cluster_min:2,cluster_max:4,cluster_spread:6,background:'generated',
    split:'train',background_block_min:2,background_block_max:8,background_level_min:1,background_level_max:1,
    seed:17,flip_x:true,flip_y:false,rotation_degrees:10,brightness_min:0.9,brightness_max:1.1,insertion:'mask',
    edges:'hard',blur_radius:0.8,blending:'replace',max_iou:0.15,noise_max:0,erode_max:0,morph_reseed:true}};
const tapeFields=[['tape-length','length'],['tape-objects','objects'],['tape-layout','layout'],['tape-jitter','jitter'],
  ['tape-cluster-min','cluster_min'],['tape-cluster-max','cluster_max'],['tape-cluster-spread','cluster_spread'],
  ['tape-bg','background'],['tape-split','split'],['tape-bg-block-min','background_block_min'],
  ['tape-bg-block-max','background_block_max'],['tape-bg-level-min','background_level_min'],['tape-bg-level-max','background_level_max'],
  ['tape-seed','seed'],['tape-flip-x','flip_x'],['tape-flip-y','flip_y'],['tape-angle','rotation_degrees'],
  ['tape-bright-min','brightness_min'],['tape-bright-max','brightness_max'],['tape-insertion','insertion'],['tape-edges','edges'],
  ['tape-blur','blur_radius'],['tape-blend','blending'],['tape-iou','max_iou'],['tape-noise','noise_max'],['tape-erode','erode_max'],
  ['tape-reseed','morph_reseed']];
const tapeLayouts={random:'Случайно по всей ленте',even:'Равномерно с разбросом',clusters:'Группы рядом друг с другом'};
const tapeClamp=(value,low,high)=>Math.max(low,Math.min(high,value));

function tapeMessage(text,error=false) {
  tapeUI.message=text;
  if($('tape-message')) {$('tape-message').textContent=text;$('tape-message').classList.toggle('error',error);}
}
async function tapeAction(action) {
  const ui=tapeUI;if(ui.busy)return;
  ui.busy=true;tapeControls();
  try {await action();} catch(error) {tapeMessage(error.message,true);}
  finally {ui.busy=false;tapeControls();}
}
function openTape() {
  const ui=tapeUI;
  if(!ui.busy) {
    const previous=ui.pool.length;
    ui.pool=synBankEntries();
    for(let index=previous;index<ui.pool.length;index++)ui.poolSelected.add(index);
    for(const index of [...ui.poolSelected])if(index>=ui.pool.length)ui.poolSelected.delete(index);
    if(ui.addIndex>=ui.pool.length)ui.addIndex=-1;
    if(ui.addIndex<0&&ui.pool.length)ui.addIndex=0;
    ui.directory=syntheticUI.directory;
    renderTape();
    if(ui.addIndex>=0)tapePrepareAdd();
  }
  if(!$('tape-dialog').open)$('tape-dialog').showModal();
}
// The split of a tape: of the generated tape, else the one chosen in the rules.
const tapeSplit = () => tapeUI.tapeSplit||tapeUI.rules.split;
// Pool entries a tape of its split can take: of that split and not stubs cut by the frame edge.
// The pool of a new tape follows the split of the rules; objects added by hand, the split of the tape.
const tapeUsable = (object,split=tapeSplit()) => object.source.split===split&&!object.clipped;
function tapePoolNote(object) {
  if(object.clipped)return 'обрубок: не используется';
  return object.source.split!==tapeUI.rules.split?`не из ${tapeUI.rules.split}: не используется`:'';
}

function renderTape() {
  const ui=tapeUI,r=ui.rules;
  $('tape-dialog').innerHTML=`<div class="autogen-heading"><h2 id="tape-title">Лента (запись)</h2><button id="tape-close" aria-label="Закрыть ленту">Закрыть</button></div>
    <p>Лента — длинная запись датчиков: ${synSensor().channels} каналов в ширину, строки — последовательные опросы. Выберите объекты из набора, способ расположения, число объектов и длину в строках, сгенерируйте ленту, затем правьте её: добавляйте объекты из набора, удаляйте и перемещайте. Сохраняются CSV с сигналом, CSV с разметкой и JSON с описанием.</p>
    <div id="tape-message" class="synthetic-message" role="status" aria-live="polite"></div>
    <div id="tape-pool" class="autogen-pool"></div>
    <div class="autogen-layout"><section class="card"><h3>Правила ленты</h3><div class="autogen-rules">
      <h4>Лента</h4>
      ${synNumber('tape-length','Длина, строк',r.length,40,20000,10)}
      ${synNumber('tape-objects','Количество объектов',r.objects,0,1000,1)}
      <label>Функция расположения <select id="tape-layout">${Object.entries(tapeLayouts).map(([key,name])=>`<option value="${key}">${name}</option>`).join('')}</select></label>
      ${synNumber('tape-jitter','Разброс (доля полушага)',r.jitter,0,1,0.05)}
      ${synNumber('tape-cluster-min','Группа от, объектов',r.cluster_min,1,64,1)}
      ${synNumber('tape-cluster-max','Группа до, объектов',r.cluster_max,1,64,1)}
      ${synNumber('tape-cluster-spread','Разброс группы ± строк',r.cluster_spread,0,200,1)}
      ${synNumber('tape-seed','Seed',r.seed,0,4294967295,1)}
      <label>Часть набора <select id="tape-split">${synSplitOptions()}</select></label>
      <p class="muted">Объекты, пустые каналы фона и сохранение — только из этой части. Проверочная лента — из test.</p>
      <h4>Фон</h4>
      <label>Фон ленты <select id="tape-bg"><option value="generated">Из пустых каналов реальных кадров</option><option value="blank">Пустой (нули)</option></select></label>
      ${synNumber('tape-bg-block-min','Каналов в блоке от',r.background_block_min,1,synSensor().channels,1)}
      ${synNumber('tape-bg-block-max','Каналов в блоке до',r.background_block_max,1,synSensor().channels,1)}
      ${synNumber('tape-bg-level-min','Уровень фона × от',r.background_level_min,0,3,0.05)}
      ${synNumber('tape-bg-level-max','Уровень фона × до',r.background_level_max,0,3,0.05)}
      <h4>Объекты</h4>
      <p class="muted">Объекты набора берутся по очереди, сначала неиспользованные: повторы появляются, только когда на ленте больше объектов, чем в выбранном наборе.</p>
      <label>Новые несовершенства на каждую вставку <input id="tape-reseed" type="checkbox"></label>
      <label>Случайное отражение по X <input id="tape-flip-x" type="checkbox"></label>
      <label>Случайное отражение по Y <input id="tape-flip-y" type="checkbox"></label>
      ${synNumber('tape-angle','Поворот ± °',r.rotation_degrees,0,180,1)}
      ${synNumber('tape-bright-min','Яркость × от',r.brightness_min,0,4,0.05)}
      ${synNumber('tape-bright-max','Яркость × до',r.brightness_max,0,4,0.05)}
      ${synNumber('tape-noise','Шум пикселей до',r.noise_max,0,1,0.01)}
      ${synNumber('tape-erode','Выедание краёв до',r.erode_max,0,1,0.05)}
      <label>Перенос <select id="tape-insertion"><option value="box">Весь бокс</option><option value="mask">Маска объекта</option></select></label>
      <label>Края вставки <select id="tape-edges"><option value="hard">Обычная вставка</option><option value="blur">Блюр краёв</option></select></label>
      ${synNumber('tape-blur','Радиус блюра, px',r.blur_radius,0.1,3,0.1)}
      <label>Наложение <select id="tape-blend"><option value="replace">Replace</option><option value="max">Max blending</option><option value="additive">Additive blending</option></select></label>
      ${synNumber('tape-iou','Макс. IoU',r.max_iou,0,1,0.01)}
      <button id="tape-generate" class="primary">Сгенерировать ленту</button>
    </div><p>Если для объекта не нашлось свободного места, он пропускается: в сообщении будет указано, сколько размещено из заказанного. Объекты не пересекаются сильнее макс. IoU.</p></section>
    <section><h3>Лента</h3><div class="synthetic-fields tape-nav">
        <span id="tape-position" class="muted"></span>
        ${synNumber('tape-rows','Окно, строк',ui.rows,20,100,5)}
        ${synNumber('tape-goto','Перейти к строке',0,0,20000,1)}<button id="tape-go">Перейти</button></div>
      <div class="tape-view">
        <canvas id="tape-map" width="${2*synSensor().channels}" height="${tapeMapHeight}" class="tape-map" aria-label="Обзор всей ленты: клик переносит окно"></canvas>
        <canvas id="tape-main" class="tape-main" aria-label="Окно ленты: клик выбирает объект, перетаскивание перемещает"></canvas>
        <div class="tape-side"><h4>Объекты на ленте</h4><div id="tape-list" class="tape-list" role="group" aria-label="Объекты на ленте"></div></div>
      </div>
      <div id="tape-stats" class="synthetic-stats"></div>
      <h4>Правка</h4>
      <div class="synthetic-fields"><label>Добавить из набора <select id="tape-add-select"></select></label>
        <button id="tape-place">Разместить кликом</button><span id="tape-add-info" class="muted"></span></div>
      <div class="synthetic-toolbar"><button id="tape-undo">Undo</button><button id="tape-redo">Redo</button><button id="tape-delete">Удалить выбранный</button></div>
      <div class="synthetic-fields">${synNumber('tape-x','X (канал)',0,0,39,1)}${synNumber('tape-y','Y (строка)',0,0,20000,1)}<button id="tape-position-apply">Применить координаты</button></div>
      <div class="synthetic-fields">${synNumber('tape-bg-seed','Seed фона',0,0,4294967295,1)}<button id="tape-bg-new">Новый фон (seed + 1)</button>
        ${synNumber('tape-edit-iou','Макс. IoU ленты',0.15,0,1,0.01)}</div>
      <p>Клик по объекту выделяет его, перетаскивание перемещает; колесо мыши и миникарта листают ленту. Объекты не могут пересекаться сильнее макс. IoU и выходить за ленту.</p>
    </section></div>
    <div class="autogen-save"><div class="synthetic-fields">
      <label class="path">Отдельная директория <input id="tape-directory" type="text" value="${esc(ui.directory)}"></label>
      <span id="tape-save-split"></span>
      <label>Имя (необязательно) <input id="tape-name" type="text" value="${esc(ui.name)}" placeholder="tape_…"></label>
      <label>Значения <select id="tape-values"><option value="levels">Уровни 0–255 (как в PNG)</option><option value="intensity">Интенсивность 0–1</option></select></label>
      <label><input id="tape-header" type="checkbox"> Строка заголовка в CSV</label>
      <button id="tape-save">Сохранить ленту</button></div>
      <p id="tape-provenance"></p><output id="tape-output" class="synthetic-output"></output></div>`;
  for(const [id,key] of tapeFields) {
    const node=$(id);
    if(node.type==='checkbox')node.checked=r[key];else node.value=r[key];
    node.onchange=()=>{tapeReadRules();if(id==='tape-split'){tapePool();tapeAddOptions();}tapeControls();};
  }
  $('tape-values').value=ui.values;$('tape-header').checked=ui.header;
  tapePool();tapeAddOptions();
  tapeBind();
  $('tape-output').textContent=ui.output;
  tapeMessage(ui.message||'Выберите правила и сгенерируйте ленту.');
  tapeDraw();tapeList();tapeControls();
}
function tapeReadRules() {
  const ui=tapeUI;
  for(const [id,key] of tapeFields) {
    const node=$(id);
    ui.rules[key]=node.type==='checkbox'?node.checked:node.type==='number'?Number(node.value):node.value;
  }
}

/* ---- loading a preview: pixels, minimap ---- */
function tapeLoad(tape,preview) {
  const ui=tapeUI,bytes=atob(preview.levels_base64),levels=new Uint8Array(bytes.length);
  for(let i=0;i<bytes.length;i++)levels[i]=bytes.charCodeAt(i);
  ui.tape=tape;ui.preview=preview;ui.levels=levels;
  ui.selected=Math.min(ui.selected,tape.objects.length-1);
  // The overview keeps, for every pixel row of the minimap, the brightest value of the tape rows it covers.
  // Two map pixels per channel.
  const length=preview.length,channels=preview.channels,width=2*channels,base=new ImageData(width,tapeMapHeight);
  for(let out=0;out<tapeMapHeight;out++) {
    const first=Math.floor(out*length/tapeMapHeight),last=Math.max(first+1,Math.floor((out+1)*length/tapeMapHeight));
    for(let channel=0;channel<channels;channel++) {
      let best=0;
      for(let row=first;row<last&&row<length;row++)best=Math.max(best,levels[row*channels+channel]);
      const gray=Math.min(255,best*2);
      for(let k=0;k<2;k++) {const at=(out*width+channel*2+k)*4;base.data[at]=base.data[at+1]=base.data[at+2]=gray;base.data[at+3]=255;}
    }
  }
  ui.mapBase=base;
}
function tapeDraw() {
  const ui=tapeUI,canvas=$('tape-main');if(!canvas)return;
  const ctx=canvas.getContext('2d');
  const channels=synSensor().channels;
  canvas.width=channels*tapeCell;canvas.height=ui.rows*tapeCell;
  const map=$('tape-map'),mapContext=map.getContext('2d');
  mapContext.clearRect(0,0,map.width,map.height);
  if(!ui.levels) {
    ctx.fillStyle='#182332';ctx.fillRect(0,0,canvas.width,canvas.height);
    $('tape-position').textContent='Ленты пока нет';
    $('tape-stats').textContent='Сгенерируйте ленту.';
    return;
  }
  const length=ui.preview.length;
  ui.top=tapeClamp(ui.top,0,Math.max(0,length-ui.rows));
  const rows=Math.min(ui.rows,length-ui.top),image=new ImageData(channels,rows);
  for(let i=0;i<channels*rows;i++) {
    const gray=ui.levels[ui.top*channels+i];
    image.data[i*4]=image.data[i*4+1]=image.data[i*4+2]=gray;image.data[i*4+3]=255;
  }
  const part=document.createElement('canvas');part.width=channels;part.height=rows;part.getContext('2d').putImageData(image,0,0);
  ctx.imageSmoothingEnabled=false;ctx.drawImage(part,0,0,channels*tapeCell,rows*tapeCell);
  ctx.font='14px system-ui';
  ui.preview.boxes_pixels_xyxy.forEach((box,index)=>{
    if(box[3]<=ui.top||box[1]>=ui.top+rows)return;
    const selected=index===ui.selected;
    ctx.strokeStyle=ctx.fillStyle=selected?'#fff177':'#ffb571';ctx.lineWidth=selected?4:2;
    ctx.strokeRect(box[0]*tapeCell,(box[1]-ui.top)*tapeCell,(box[2]-box[0])*tapeCell,(box[3]-box[1])*tapeCell);
    ctx.fillText(String(index+1),box[0]*tapeCell+3,Math.max(15,(box[1]-ui.top)*tapeCell+14));
  });
  if(ui.drag) {
    const box=ui.preview.boxes_pixels_xyxy[ui.drag.index],obj=ui.tape.objects[ui.drag.index];
    ctx.strokeStyle='#fff177';ctx.lineWidth=3;ctx.setLineDash([8,5]);
    ctx.strokeRect((box[0]-obj.x+ui.drag.x)*tapeCell,(box[1]-obj.y+ui.drag.y-ui.top)*tapeCell,(box[2]-box[0])*tapeCell,(box[3]-box[1])*tapeCell);
    ctx.setLineDash([]);
  }
  mapContext.putImageData(ui.mapBase,0,0);
  const scale=tapeMapHeight/length;
  mapContext.strokeStyle='#ffb571';mapContext.lineWidth=1;
  for(const box of ui.preview.boxes_pixels_xyxy)mapContext.strokeRect(box[0]*2,box[1]*scale,(box[2]-box[0])*2,Math.max(2,(box[3]-box[1])*scale));
  mapContext.strokeStyle='#80c8ff';mapContext.lineWidth=2;
  mapContext.strokeRect(1,ui.top*scale,78,Math.max(4,rows*scale));
  $('tape-position').textContent=`Строки ${ui.top}–${ui.top+rows-1} из ${length}`;
  const s=ui.preview.statistics;
  $('tape-stats').innerHTML=`<span>Длина <strong>${s.length}</strong> строк</span><span>Объектов <strong>${s.total_objects}</strong></span><span>Макс. IoU <strong>${s.max_iou.toFixed(3)}</strong></span><span>Мин. расстояние <strong>${s.min_distance===null?'—':s.min_distance.toFixed(2)+' px'}</strong></span>`;
  const splits=ui.preview.splits||[],own=splits.every(split=>split===tapeSplit());
  $('tape-provenance').textContent=own?`Объекты и фон ленты из ${tapeSplit()}: лента сохраняется только в ${tapeSplit()}.`:`В ленте источники из ${splits.join(', ')}: всё должно быть из ${tapeSplit()}, такую ленту сохранить нельзя.`;
  $('tape-provenance').classList.toggle('error',!own);
}
function tapePool() {
  const ui=tapeUI;
  $('tape-pool').innerHTML=ui.pool.length?ui.pool.map((object,index)=>{
    const note=tapePoolNote(object);
    return `<label class="${note?'muted':''}"><input id="tape-pool-${index}" type="checkbox" ${ui.poolSelected.has(index)&&!note?'checked':''} ${note?'disabled':''}> <span>${index+1} · ${esc(object.source.split+' / '+object.source.id)}<br>${esc(synObjectLabel(object))}${note?'<br>'+esc(note):''}</span></label>`;
  }).join(''):'<p class="muted">Набор объектов пуст. Добавьте рамки в редакторе или объекты из генератора объектов.</p>';
  ui.pool.forEach((_,index)=>$(`tape-pool-${index}`).onchange=()=>{
    if($(`tape-pool-${index}`).checked)ui.poolSelected.add(index);else ui.poolSelected.delete(index);
    tapeControls();
  });
}
function tapeAddOptions() {
  const ui=tapeUI;
  $('tape-add-select').innerHTML=ui.pool.map((object,index)=>`<option value="${index}" ${tapeUsable(object)?'':'disabled'}>${index+1} · ${esc(object.source.id)} · ${esc(synObjectLabel(object))}</option>`).join('');
  if(ui.addIndex>=0&&!tapeUsable(ui.pool[ui.addIndex]||{source:{}})) {
    ui.addIndex=ui.pool.findIndex(tapeUsable);
    if(ui.addIndex>=0)tapePrepareAdd();else ui.addPrepared=null;
  }
  if(ui.addIndex>=0)$('tape-add-select').value=ui.addIndex;
}
const tapeChosenPool = () => [...tapeUI.poolSelected].sort((a,b)=>a-b).map(index=>tapeUI.pool[index]).filter(object=>tapeUsable(object,tapeUI.rules.split));
function tapeList() {
  const ui=tapeUI,list=$('tape-list');if(!list)return;
  list.innerHTML=ui.tape?ui.tape.objects.map((obj,index)=>`<button data-index="${index}" class="${index===ui.selected?'active':''}">${index+1}: ${esc(obj.source.id)} · ${esc(synObjectLabel(obj))} @ ${obj.x}, ${obj.y}</button>`).join(''):'';
  list.querySelectorAll('button').forEach(button=>button.onclick=()=>tapeSelect(Number(button.dataset.index),true));
  const active=list.querySelector('.active');
  if(active)list.scrollTop=Math.max(0,active.offsetTop-list.clientHeight/2);
}
function tapeSelect(index,reveal=false) {
  const ui=tapeUI;
  ui.selected=index;ui.placing=false;
  const obj=ui.tape?.objects[index];
  if(obj) {
    $('tape-x').value=obj.x;$('tape-y').value=obj.y;
    const box=ui.preview.boxes_pixels_xyxy[index];
    if(reveal&&(box[1]<ui.top||box[3]>ui.top+ui.rows))ui.top=Math.round(box[1]-ui.rows/2);
  }
  tapeDraw();tapeList();tapeControls();
}
function tapeControls() {
  if(!$('tape-generate'))return;
  const ui=tapeUI,r=ui.rules;
  document.querySelectorAll('#tape-dialog input, #tape-dialog select, #tape-dialog button').forEach(node=>node.disabled=ui.busy);
  $('tape-close').disabled=false;
  const has=!!ui.tape,pool=ui.pool.length>0;
  $('tape-generate').disabled=ui.busy||!tapeChosenPool().length;
  ui.pool.forEach((object,index)=>$(`tape-pool-${index}`).disabled=ui.busy||!tapeUsable(object,r.split));
  $('tape-save-split').textContent=has?`Часть: ${tapeSplit()}`:'';
  $('tape-jitter').disabled=ui.busy||r.layout!=='even';
  for(const id of ['tape-cluster-min','tape-cluster-max','tape-cluster-spread'])$(id).disabled=ui.busy||r.layout!=='clusters';
  for(const id of ['tape-bg-block-min','tape-bg-block-max','tape-bg-level-min','tape-bg-level-max'])$(id).disabled=ui.busy||r.background!=='generated';
  $('tape-blur').disabled=ui.busy||r.edges!=='blur';
  $('tape-place').disabled=ui.busy||!has||!ui.addPrepared;$('tape-place').classList.toggle('active',ui.placing);
  $('tape-add-select').disabled=ui.busy||!pool;
  $('tape-undo').disabled=ui.busy||!ui.undo.length;$('tape-redo').disabled=ui.busy||!ui.redo.length;
  $('tape-delete').disabled=$('tape-position-apply').disabled=ui.busy||ui.selected<0;
  $('tape-x').disabled=$('tape-y').disabled=ui.busy||ui.selected<0;
  $('tape-go').disabled=$('tape-goto').disabled=$('tape-rows').disabled=ui.busy||!has;
  $('tape-bg-seed').disabled=$('tape-bg-new').disabled=ui.busy||!has||ui.tape?.background.kind!=='generated';
  $('tape-edit-iou').disabled=ui.busy||!has;
  $('tape-save').disabled=ui.busy||!has||!(ui.preview?.splits||[]).every(split=>split===tapeSplit());
  if(has) {
    if(ui.tape.background.kind==='generated')$('tape-bg-seed').value=ui.tape.background.seed;
    $('tape-edit-iou').value=ui.tape.max_iou;
  }
}

/* ---- the API ---- */
async function tapeCommit(candidate,history=true) {
  const ui=tapeUI,preview=await synPost('tape-preview',{tape:candidate});
  if(history) {ui.undo.push(synCopy(ui.tape));if(ui.undo.length>30)ui.undo.shift();ui.redo=[];}
  tapeLoad(candidate,preview);
  tapeMessage('Лента проверена: рамки, границы и IoU корректны.');
  tapeDraw();tapeList();
}
async function tapeHistory(redo) {
  const ui=tapeUI,from=redo?ui.redo:ui.undo,to=redo?ui.undo:ui.redo,candidate=from.at(-1);
  if(!candidate)return;
  const previous=synCopy(ui.tape);
  await tapeCommit(synCopy(candidate),false);
  from.pop();to.push(previous);ui.placing=false;
  tapeList();
}
async function tapeGenerate() {
  const ui=tapeUI;tapeReadRules();ui.tapeSplit=null;
  const pool=tapeChosenPool().map(({clipped,...entry})=>entry);
  tapeMessage('Генерация ленты…');
  const result=await synPost('tape-generate',{pool,rules:ui.rules});
  ui.tapeSplit=result.rules.generator.split;tapeAddOptions();
  ui.undo=[];ui.redo=[];ui.selected=-1;ui.top=0;ui.placing=false;ui.output='';
  tapeLoad(result.tape,result.preview);
  let message=`Размещено ${result.placed} из ${result.requested} объектов.`;
  const used=result.usage.filter((_,index)=>index<pool.length);
  if(used.length&&result.placed)message+=`\nКаждый объект набора использован от ${Math.min(...used)} до ${Math.max(...used)} раз.`;
  if(result.notes.length)message+='\n'+result.notes.join('\n');
  if(Object.keys(result.rejected).length)message+='\nОтклонённые попытки: '+Object.entries(result.rejected).map(([reason,count])=>`${reason} (${count})`).join('; ');
  tapeMessage(message);
  $('tape-output').textContent='';tapeDraw();tapeList();
}
async function tapePrepareAdd() {
  const ui=tapeUI,entry=ui.pool[ui.addIndex],token=++ui.addToken;
  ui.addPrepared=null;tapeControls();
  if(!entry||!tapeUsable(entry))return;
  try {
    const {clipped,...body}=entry;
    const prepared=await synPost('tape-prepare',body);
    if(token===ui.addToken)ui.addPrepared=prepared;
  } catch(error) {if(token===ui.addToken)tapeMessage(error.message,true);}
  tapeControls();
}
async function tapeSave() {
  const ui=tapeUI;
  ui.directory=$('tape-directory').value;ui.name=$('tape-name').value.trim();
  ui.values=$('tape-values').value;ui.header=$('tape-header').checked;
  tapeMessage('Проверка и сохранение ленты…');
  const result=await synPost('tape-save',{tape:ui.tape,expected_digest:ui.preview.digest,directory:ui.directory,
    split:tapeSplit(),name:ui.name,values:ui.values,header:ui.header});
  ui.output=`Сохранено:\n${result.signal}\n${result.labels}\n${result.provenance}`;
  $('tape-output').textContent=ui.output;
  tapeMessage('Лента сохранена: CSV сигнала, CSV разметки и JSON описания.');
}

/* ---- pointer, wheel and the rest of the bindings ---- */
function tapePoint(event) {
  const canvas=$('tape-main'),rect=canvas.getBoundingClientRect();
  return {x:(event.clientX-rect.left)*synSensor().channels/rect.width,y:tapeUI.top+(event.clientY-rect.top)*tapeUI.rows/rect.height};
}
function tapeHit(point) {
  const boxes=tapeUI.preview.boxes_pixels_xyxy;
  for(let i=boxes.length-1;i>=0;i--) {const b=boxes[i];if(point.x>=b[0]&&point.x<=b[2]&&point.y>=b[1]&&point.y<=b[3])return i;}
  return -1;
}
function tapeMove(event) {
  const ui=tapeUI,rect=$('tape-map').getBoundingClientRect();
  ui.top=Math.round((event.clientY-rect.top)/rect.height*ui.preview.length-ui.rows/2);
  tapeDraw();
}
function tapeBind() {
  const ui=tapeUI,main=$('tape-main'),map=$('tape-map');
  $('tape-close').onclick=()=>$('tape-dialog').close();
  $('tape-generate').onclick=()=>tapeAction(tapeGenerate);
  $('tape-add-select').onchange=()=>{ui.addIndex=Number($('tape-add-select').value);tapePrepareAdd();};
  $('tape-place').onclick=()=>{ui.placing=!ui.placing;if(ui.placing){ui.selected=-1;tapeMessage('Кликните центр нового объекта на ленте.');}tapeDraw();tapeList();tapeControls();};
  $('tape-undo').onclick=()=>tapeAction(()=>tapeHistory(false));$('tape-redo').onclick=()=>tapeAction(()=>tapeHistory(true));
  $('tape-delete').onclick=()=>tapeAction(async()=>{
    if(ui.selected<0)return;
    const candidate=synCopy(ui.tape);candidate.objects.splice(ui.selected,1);
    await tapeCommit(candidate);ui.selected=-1;tapeDraw();tapeList();
  });
  $('tape-position-apply').onclick=()=>tapeAction(async()=>{
    const candidate=synCopy(ui.tape);
    candidate.objects[ui.selected].x=Number($('tape-x').value);candidate.objects[ui.selected].y=Number($('tape-y').value);
    await tapeCommit(candidate);
  });
  $('tape-bg-seed').onchange=()=>tapeAction(async()=>{const candidate=synCopy(ui.tape);candidate.background.seed=Number($('tape-bg-seed').value);await tapeCommit(candidate);});
  $('tape-bg-new').onclick=()=>tapeAction(async()=>{const candidate=synCopy(ui.tape);candidate.background.seed+=1;await tapeCommit(candidate);});
  $('tape-edit-iou').onchange=()=>tapeAction(async()=>{
    const candidate=synCopy(ui.tape);candidate.max_iou=Number($('tape-edit-iou').value);
    try {await tapeCommit(candidate);} finally {$('tape-edit-iou').value=ui.tape.max_iou;}
  });
  $('tape-rows').onchange=()=>{ui.rows=tapeClamp(Number($('tape-rows').value)||60,20,100);tapeDraw();};
  $('tape-go').onclick=()=>{ui.top=Number($('tape-goto').value)||0;tapeDraw();};
  $('tape-values').onchange=()=>ui.values=$('tape-values').value;
  $('tape-save').onclick=()=>tapeAction(tapeSave);
  main.addEventListener('wheel',event=>{
    if(!ui.levels)return;
    event.preventDefault();ui.top+=Math.sign(event.deltaY)*3;tapeDraw();
  },{passive:false});
  map.onpointerdown=event=>{if(!ui.levels||ui.busy)return;map.setPointerCapture(event.pointerId);map.dragging=true;tapeMove(event);};
  map.onpointermove=event=>{if(map.dragging)tapeMove(event);};
  map.onpointerup=map.onpointercancel=()=>{map.dragging=false;};
  main.onpointerdown=event=>{
    if(ui.busy||!ui.levels)return;
    const point=tapePoint(event);
    if(ui.placing) {
      const prepared=ui.addPrepared;if(!prepared)return;
      tapeAction(async()=>{
        const entry=ui.pool[ui.addIndex],added={source:synCopy(entry.source),box_index:entry.box_index,transform:synCopy(entry.transform),
          x:tapeClamp(Math.round(point.x-prepared.width/2),0,synSensor().channels-prepared.width),y:Math.max(0,Math.round(point.y-prepared.height/2))};
        if(entry.morph)added.morph=synCopy(entry.morph);
        const candidate=synCopy(ui.tape);candidate.objects.push(added);
        await tapeCommit(candidate);
        ui.placing=false;tapeSelect(candidate.objects.length-1);
      });
      return;
    }
    const index=tapeHit(point);
    tapeSelect(index);
    if(index<0)return;
    const obj=ui.tape.objects[index];
    ui.drag={index,dx:point.x-obj.x,dy:point.y-obj.y,x:obj.x,y:obj.y};main.setPointerCapture(event.pointerId);
  };
  main.onpointermove=event=>{
    if(!ui.drag||ui.busy)return;
    const point=tapePoint(event);
    ui.drag.x=Math.round(point.x-ui.drag.dx);ui.drag.y=Math.round(point.y-ui.drag.dy);
    $('tape-x').value=ui.drag.x;$('tape-y').value=ui.drag.y;tapeDraw();
  };
  main.onpointerup=()=>{
    if(!ui.drag)return;
    const drag=ui.drag,obj=ui.tape.objects[drag.index];ui.drag=null;
    if(obj.x===drag.x&&obj.y===drag.y){tapeDraw();return;}
    tapeAction(async()=>{
      const candidate=synCopy(ui.tape);candidate.objects[drag.index].x=drag.x;candidate.objects[drag.index].y=drag.y;
      try {await tapeCommit(candidate);} finally {tapeDraw();tapeList();}
    });
  };
  main.onpointercancel=()=>{ui.drag=null;tapeDraw();};
}
$('tape-open').onclick=openTape;
