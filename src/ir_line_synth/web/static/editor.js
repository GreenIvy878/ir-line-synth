/* Manual scene state; all authoritative pixels/annotations come from the API. */
'use strict';
const syntheticUI = {
  catalog:null, donor:null, donorSplit:'train', donorId:null, objectBank:[], activeObject:-1,
  scene:{background:null,objects:[],blending:'replace',max_iou:0.15}, preview:null,
  selected:-1, placing:false, drag:null, undo:[], redo:[], busy:false,
  sourceToken:0, message:'Выберите исходный объект и фон.',
  directory:'synthetic_data', outputSplit:'train', count:5, seed:17,
  variation:{position:3,brightness:0.1,angle:10,scale:0.1}, output:'', backgroundSeed:1
};
// Short label of a bank or scene object: a real box, or a generated object of a kind.
// A stub (a box cut by the frame edge) is marked: the generators and tapes never take it.
const synObjectLabel = object => (object.morph?`новый: ${(typeof morphKinds!=='undefined'&&morphKinds[object.morph.kind])||object.morph.kind}`:`рамка ${object.box_index+1}`)+(object.clipped?' · обрубок':'');
// Splits of the source dataset; a scene, a batch or a tape takes everything from one of them.
const synSplitNames = {train:'Train',val:'Validation',test:'Test'};
// The sensor line of the dataset (GET /api/synthetic/sensor): channels, polls per frame and the frame size.
const synSensor = () => syntheticUI.sensor;
// Canvas size attributes for a frame, its longer side `longest` pixels.
const synCanvasSize = longest => {const s=synSensor(),k=longest/Math.max(s.frame_width,s.frame_height);
  return `width="${Math.round(s.frame_width*k)}" height="${Math.round(s.frame_height*k)}"`;};
const synSplits = () => ['train','val',...(syntheticUI.catalog?.test?['test']:[])];
const synSplitOptions = () => synSplits().map(split=>`<option value="${split}">${synSplitNames[split]}</option>`).join('');
// The one split a preview may be saved into, or null when its sources are of several splits.
const synOnlySplit = preview => preview?.splits?.length===1?preview.splits[0]:null;
const synBackgroundLabel = background => !background?'не выбран':background.generated?`сгенерированный (seed ${background.generated.seed}, каналы из ${background.split})`:`${background.split} / ${background.id}`;
const synCopy = value => JSON.parse(JSON.stringify(value));
const synPost = (path,body) => request(`/api/synthetic/${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const synMounted = () => !!$('synthetic-editor');
const synTransformFields = [
  ['syn-flip-x','flip_x'],['syn-flip-y','flip_y'],['syn-use-mask','use_mask'],
  ['syn-brightness','brightness'],['syn-angle','angle'],['syn-scale','scale'],['syn-threshold','threshold'],
  ['syn-gamma','gamma'],['syn-contrast','contrast'],['syn-gradient','gradient'],['syn-gradient-angle','gradient_angle'],
  ['syn-noise','noise'],['syn-noise-seed','noise_seed'],
  ['syn-shade','shade'],['syn-shade-size','shade_size'],['syn-shade-x','shade_x'],['syn-shade-y','shade_y'],
  ['syn-erode','erode'],['syn-erode-seed','erode_seed']
];
const synDefaultTransform = () => ({flip_x:false,flip_y:false,brightness:1,angle:0,scale:1,use_mask:false,threshold:0.1,
  gamma:1,contrast:1,gradient:0,gradient_angle:0,noise:0,noise_seed:0,shade:0,shade_size:0.25,shade_x:0.5,shade_y:0.5,erode:0,erode_seed:0});
const synActiveObject = () => syntheticUI.objectBank[syntheticUI.activeObject]||null;
// The object bank as the pool of the autogenerator and of tapes.
const synBankEntries = () => syntheticUI.objectBank.map(object=>{
  const entry={source:synCopy(object.source),box_index:object.box_index,transform:synCopy(object.transform)};
  if(object.morph)entry.morph=synCopy(object.morph);
  if(object.clipped)entry.clipped=true;
  return entry;
});

function synSyncTransformFields() {
  if(!synMounted())return;
  const transform=synActiveObject()?.transform||synDefaultTransform();
  for(const [id,key] of synTransformFields) {
    if($(id).type==='checkbox')$(id).checked=transform[key];
    else $(id).value=transform[key];
  }
}
function synSelectObject(index) {
  const ui=syntheticUI;
  if(ui.busy||!ui.objectBank[index])return;
  ui.activeObject=index;ui.placing=false;
  synSyncTransformFields();synDraw();synButtons();
  const object=synActiveObject();
  if(object.error)synMessage(object.error,true);
  else synMessage(`Выбран объект ${index+1}: ${object.source.split} / ${object.source.id}, ${synObjectLabel(object)}.`);
}
function synCollectObject(boxIndex) {
  const ui=syntheticUI;
  const object={source:{split:ui.donor.split,id:ui.donor.id},box_index:boxIndex,
    transform:synDefaultTransform(),prepared:null,error:null,prepareToken:0};
  if(ui.donor.clipped?.[boxIndex])object.clipped=true;
  ui.objectBank.push(object);
  synSelectObject(ui.objectBank.length-1);
  if(object.clipped)synMessage(`Объект ${ui.objectBank.length} — обрубок: рамка у края кадра. Его можно поставить в сцену вручную, но автогенератор, ленты и генератор объектов его не берут.`);
  return synPrepare(object);
}

function synMessage(text,error=false) {
  syntheticUI.message=text;
  if(synMounted()) {$('syn-message').textContent=text;$('syn-message').classList.toggle('error',error);}
}
async function synAction(action) {
  if(syntheticUI.busy)return;
  syntheticUI.busy=true;synButtons();
  try {await action();} catch(error) {synMessage(error.message,true);}
  finally {syntheticUI.busy=false;synButtons();}
}
function synOptions(split,selected) {
  return (syntheticUI.catalog?.[split]||[]).map(row=>`<option value="${esc(row.id)}" ${row.id===selected?'selected':''}>${esc(row.id)} · ${row.objects} объектов</option>`).join('');
}
function synNumber(id,label,value,min,max,step) {
  return `<label>${label} <input id="${id}" type="number" value="${value}" min="${min}" max="${max}" step="${step}"></label>`;
}
async function renderSyntheticEditor() {
  if($('synthetic-editor'))return;
  const ui=syntheticUI;
  if(!ui.catalog) {
    $('content').innerHTML='<h2>Synthetic Editor</h2><p id="syn-loading">Загрузка train / val…</p>';
    try {[ui.sensor,ui.catalog]=await Promise.all([request('/api/synthetic/sensor'),request('/api/synthetic/catalog')]);}
    catch(error) {$('content').innerHTML=`<h2>Synthetic Editor</h2><p class="error">${esc(error.message)}</p>`;return;}
  }
  const bg=ui.scene.background;
  const transform=synActiveObject()?.transform||synDefaultTransform();
  ui.drag=null;
  $('content').innerHTML=`<div id="synthetic-editor"><h2>Synthetic Editor</h2>
    <p>Кадр ${synSensor().frame_width} × ${synSensor().frame_height} · каналов: ${synSensor().channels}, опросов в кадре: ${synSensor().frame_length} · координаты исходного изображения. Синие рамки — исходные; оранжевые — добавленные. Данные train и val доступны для просмотра.</p>
    <div id="syn-message" class="synthetic-message" role="status" aria-live="polite"></div>
    <div class="synthetic-layout">
    <section class="card synthetic-object"><h3>1. Редактирование выбранного объекта</h3><div class="synthetic-transform">
      <div class="synthetic-fields">
        <label><input id="syn-flip-x" type="checkbox" ${transform.flip_x?'checked':''}> Горизонтальное отражение</label>
        <label><input id="syn-flip-y" type="checkbox" ${transform.flip_y?'checked':''}> Вертикальное отражение</label>
        ${synNumber('syn-brightness','Яркость ×',transform.brightness,0,4,0.05)}
        ${synNumber('syn-angle','Угол °',transform.angle,-360,360,1)}
        ${synNumber('syn-scale','Масштаб ×',transform.scale,0.1,5,0.05)}
        <label><input id="syn-use-mask" type="checkbox" ${transform.use_mask?'checked':''}> Применять маску</label>
        ${synNumber('syn-threshold','Порог маски',transform.threshold,0,1,0.005)}
        <h4 class="synthetic-wide">Изменение соотношений яркости пикселей</h4>
        ${synNumber('syn-gamma','Гамма (пересборка)',transform.gamma,0.3,3,0.05)}
        ${synNumber('syn-contrast','Контраст ×',transform.contrast,0,3,0.05)}
        ${synNumber('syn-gradient','Градиент света',transform.gradient,-1,1,0.05)}
        ${synNumber('syn-gradient-angle','Направление градиента °',transform.gradient_angle,-360,360,5)}
        ${synNumber('syn-noise','Шум пикселей',transform.noise,0,1,0.01)}
        ${synNumber('syn-noise-seed','Seed шума',transform.noise_seed,0,4294967295,1)}
        <h4 class="synthetic-wide">Затенение (пыль, грязь)</h4>
        ${synNumber('syn-shade','Сила затемнения',transform.shade,0,1,0.05)}
        ${synNumber('syn-shade-size','Размер пятна (доля объекта)',transform.shade_size,0.05,1,0.05)}
        ${synNumber('syn-shade-x','Пятно X (0–1)',transform.shade_x,0,1,0.05)}
        ${synNumber('syn-shade-y','Пятно Y (0–1)',transform.shade_y,0,1,0.05)}
        <h4 class="synthetic-wide">Выедание краёв</h4>
        ${synNumber('syn-erode','Доля выбитых краевых пикселей',transform.erode,0,1,0.05)}
        ${synNumber('syn-erode-seed','Seed выедания',transform.erode_seed,0,4294967295,1)}
        <button id="syn-reset">Без изменений</button><span id="syn-patch-info" class="muted"></span>
      </div><div class="synthetic-bank"><h4>Выбранный объект</h4>
        <div class="controls"><button id="syn-prev-object" type="button" aria-label="Предыдущий объект">←</button>
          <label>№ <input id="syn-object-number" type="number" min="1" step="1"></label><span id="syn-object-total" class="muted"></span>
          <button id="syn-next-object" type="button" aria-label="Следующий объект">→</button></div>
        <p id="syn-object-status" class="muted"></p></div>
    </div></section>
    <section class="card synthetic-source"><h3>2. Исходное изображение</h3>
      <div class="controls"><label>Источник <select id="syn-source-split">${synSplitOptions()}</select></label></div>
      <select id="syn-source-id" class="synthetic-choice" aria-label="Source image"><option value="">Выберите изображение</option>${synOptions(ui.donorSplit,ui.donorId)}</select>
      <canvas id="syn-source" ${synCanvasSize(640)} class="synthetic-canvas" aria-label="Select an annotated source object"></canvas>
      <p id="syn-source-info">Клик по рамке добавляет объект в набор.</p>
      <p class="muted">Пунктирные рамки с пометкой «обрубок» срезаны краем кадра (ближе 1 ячейки к краю или пиксели объекта на крайней строке/столбце): автогенератор, ленты и генератор объектов их не используют.</p>
    </section>
    <section class="card synthetic-scene"><h3>3. Фон и собранная сцена</h3>
      <div class="controls"><label>Фон <select id="syn-bg-split">${synSplitOptions()}</select></label>
      <label>Наложение <select id="syn-blend"><option value="replace">Replace</option><option value="max">Max blending</option><option value="additive">Additive blending</option></select></label>
      ${synNumber('syn-iou','Макс. IoU',ui.scene.max_iou,0,1,0.01)}</div>
      <select id="syn-bg-id" class="synthetic-choice" aria-label="Background image"><option value="">Выберите фон</option>${synOptions(bg?.split||'train',bg?.id)}</select>
      <div class="synthetic-fields"><button id="syn-bg-generate">Сгенерировать фон</button>${synNumber('syn-bg-seed','Seed фона',ui.backgroundSeed,0,4294967295,1)}<span id="syn-bg-info" class="muted"></span></div>
      <canvas id="syn-scene" ${synCanvasSize(640)} class="synthetic-canvas" aria-label="Position and move scene objects"></canvas>
      <div class="synthetic-toolbar"><button id="syn-place">Разместить объект кликом</button><button id="syn-undo">Undo</button><button id="syn-redo">Redo</button><button id="syn-delete">Удалить выбранный</button></div>
      <div class="synthetic-fields">${synNumber('syn-x','X',0,0,synSensor().frame_width-1,1)}${synNumber('syn-y','Y',0,0,synSensor().frame_height-1,1)}<button id="syn-position">Применить координаты</button></div>
      <div id="syn-objects" class="synthetic-objects"></div><div id="syn-stats" class="synthetic-stats"></div>
      <p>Клик выделяет добавленный объект; перетаскивание перемещает его. X/Y — левый верхний угол. Смена фона сохраняет добавленные объекты и проверяет размещение.</p>
    </section>
    <section class="card synthetic-export"><h3>4. Сохранение и вариации</h3>
      <div class="synthetic-fields"><label class="path">Отдельная директория <input type="text" id="syn-directory" value="${esc(ui.directory)}"></label>
      <label>Набор <select id="syn-output-split">${synSplitOptions()}</select></label><button id="syn-save">Сохранить сцену</button></div>
      <p id="syn-provenance"></p><div class="synthetic-fields">
      ${synNumber('syn-count','Вариаций N',ui.count,1,100,1)}${synNumber('syn-seed','Seed',ui.seed,0,4294967295,1)}
      ${synNumber('syn-var-position','Сдвиг ± px',ui.variation.position,0,Math.max(synSensor().frame_width,synSensor().frame_height),1)}
      ${synNumber('syn-var-brightness','Яркость ± доля',ui.variation.brightness,0,1,0.05)}
      ${synNumber('syn-var-angle','Угол ± °',ui.variation.angle,0,180,1)}
      ${synNumber('syn-var-scale','Масштаб ± доля',ui.variation.scale,0,0.9,0.05)}
      <button id="syn-variations">Создать и сохранить вариации</button></div>
      <p>Сохранение: 8-битный PNG, как реальные кадры, + YOLO .txt + происхождение. Файл data.yaml содержит настройки загрузки. Каждая вариация изменяет только добавленные объекты и проходит проверку границ и IoU.</p>
      <output id="syn-output" class="synthetic-output"></output>
    </section></div></div>`;
  $('syn-source-split').value=ui.donorSplit;$('syn-bg-split').value=bg?.split||'train';
  $('syn-blend').value=ui.scene.blending;$('syn-output-split').value=ui.outputSplit;
  synMessage(ui.message);synBind();synDraw();synButtons();$('syn-output').textContent=ui.output;
}

function synCanvas(id,matrix,boxes=[],selected=-1,original=boxes.length,mask=null,clipped=[]) {
  const canvas=$(id);if(!canvas)return;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!matrix)return;
  const height=matrix.length,width=matrix[0].length,sx=canvas.width/width,sy=canvas.height/height;
  matrix.forEach((row,y)=>row.forEach((value,x)=>{
    const gray=Math.round(255*Math.max(0,Math.min(1,value)));
    ctx.fillStyle=mask&&!mask[y][x]?((x+y)%2?'#26374b':'#182332'):`rgb(${gray},${gray},${gray})`;
    ctx.fillRect(x*sx,y*sy,sx,sy);
  }));
  boxes.forEach((box,index)=>{
    ctx.strokeStyle=index===selected?'#fff177':clipped[index]?'#9aa6b2':index<original?'#80c8ff':'#ffb571';ctx.lineWidth=index===selected?4:2;
    ctx.setLineDash(clipped[index]?[6,4]:[]);
    ctx.strokeRect(box[0]*sx,box[1]*sy,(box[2]-box[0])*sx,(box[3]-box[1])*sy);
    ctx.setLineDash([]);
    ctx.font='14px system-ui';ctx.fillStyle=ctx.strokeStyle;ctx.fillText(String(index+1)+(clipped[index]?' обрубок':''),box[0]*sx+3,Math.max(15,box[1]*sy+14));
  });
}
function synDrawObjectBank() {
  // Only the selector is drawn: thousands of tiles with canvases hang the page.
  const ui=syntheticUI,count=ui.objectBank.length,object=synActiveObject();
  const number=$('syn-object-number');
  number.max=String(Math.max(count,1));number.value=object?String(ui.activeObject+1):'';
  number.disabled=ui.busy||!count;
  $('syn-object-total').textContent=count?`из ${count}`:'';
  $('syn-prev-object').disabled=ui.busy||ui.activeObject<=0;
  $('syn-next-object').disabled=ui.busy||ui.activeObject>=count-1;
  number.onchange=()=>{const index=Math.round(Number(number.value))-1;if(index>=0&&index<count)synSelectObject(index);else number.value=object?String(ui.activeObject+1):'';};
  $('syn-prev-object').onclick=()=>synSelectObject(ui.activeObject-1);
  $('syn-next-object').onclick=()=>synSelectObject(ui.activeObject+1);
  const prepared=object?.prepared,status=$('syn-object-status');
  status.textContent=!count?'Кликните по рамкам на исходных изображениях, чтобы собрать набор.':!object?'':object.error?'Ошибка преобразования':prepared?`${synObjectLabel(object)} · ${prepared.width} × ${prepared.height} px`:'Подготовка…';
  status.classList.toggle('error',!!object?.error);
  $('syn-patch-info').textContent=object?`Выбран ${ui.activeObject+1} / ${ui.objectBank.length} · ${object.source.split} / ${object.source.id}`:'Выберите рамку на исходном изображении.';
}
function synDraw() {
  if(!synMounted())return;
  const ui=syntheticUI;
  const active=synActiveObject();
  const donorIndex=active&&ui.donor&&active.source.split===ui.donor.split&&active.source.id===ui.donor.id?active.box_index:-1;
  const donorBoxes=ui.donor?.boxes_pixels_xyxy||[];
  synCanvas('syn-source',ui.donor?.matrix,donorBoxes,donorIndex,donorBoxes.length,null,ui.donor?.clipped||[]);
  synDrawObjectBank();
  const p=ui.preview,original=p?p.statistics.total_objects-p.statistics.added_objects:0;
  synCanvas('syn-scene',p?.matrix,p?.boxes_pixels_xyxy||[],ui.selected<0?-1:original+ui.selected,original);
  $('syn-source-info').textContent=ui.donor?`${ui.donor.split} / ${ui.donor.id} · ${ui.donor.labels.length} объектов. Клик по рамке добавит её в набор (${ui.objectBank.length}).`:'Клик по рамке добавляет объект в набор.';
  if(p) {
    const s=p.statistics;
    $('syn-stats').innerHTML=`<span>Всего <strong>${s.total_objects}</strong></span><span>Новых <strong>${s.added_objects}</strong></span><span>Макс. IoU <strong>${s.max_iou.toFixed(3)}</strong></span><span>Мин. расстояние <strong>${s.min_distance===null?'—':s.min_distance.toFixed(2)+' px'}</strong></span>`;
    const only=synOnlySplit(p);
    $('syn-provenance').textContent=only?`Все объекты и фон из ${only}: сцену можно сохранить только в ${only}.`:`В сцене источники из ${(p.splits||[]).join(', ')}: объекты и фон должны быть из одной части, такую сцену сохранить нельзя.`;
    $('syn-provenance').classList.toggle('error',!only);
    if(only)ui.outputSplit=only;
  } else {$('syn-stats').textContent='Выберите фон для предпросмотра.';$('syn-provenance').textContent='';}
  $('syn-output-split').value=ui.outputSplit;
  synLockOutputSplit();
  $('syn-objects').innerHTML=ui.scene.objects.map((obj,index)=>`<button data-object="${index}" class="${ui.selected===index?'active':''}">${index+1}: ${esc(obj.source.split+'/'+obj.source.id)}${obj.morph?' · '+esc(synObjectLabel(obj)):''} @ ${obj.x}, ${obj.y}</button>`).join('');
  $('syn-bg-info').textContent=`Фон: ${synBackgroundLabel(ui.scene.background)}. Сгенерированный фон собирается из пустых каналов реальных кадров выбранного набора.`;
  $('syn-objects').querySelectorAll('button').forEach(button=>button.onclick=()=>{ui.selected=Number(button.dataset.object);ui.placing=false;synDraw();synButtons();});
  if(ui.selected>=0&&ui.scene.objects[ui.selected]) {$('syn-x').value=ui.scene.objects[ui.selected].x;$('syn-y').value=ui.scene.objects[ui.selected].y;}
  $('syn-scene').classList.toggle('placing',ui.placing);
}
function synButtons() {
  if(!synMounted())return;
  const ui=syntheticUI;
  document.querySelectorAll('#synthetic-editor button, #synthetic-editor input, #synthetic-editor select').forEach(node=>node.disabled=ui.busy);
  const active=synActiveObject();
  for(const [id] of synTransformFields)$(id).disabled=ui.busy||!active;
  $('syn-reset').disabled=ui.busy||!active;
  $('syn-place').disabled=ui.busy||!active?.prepared||!ui.preview;$('syn-place').classList.toggle('active',ui.placing);
  $('syn-undo').disabled=ui.busy||!ui.undo.length;$('syn-redo').disabled=ui.busy||!ui.redo.length;
  $('syn-delete').disabled=$('syn-position').disabled=ui.busy||ui.selected<0;
  const only=synOnlySplit(ui.preview);
  $('syn-save').disabled=ui.busy||!ui.preview||!only;
  $('syn-variations').disabled=ui.busy||!ui.preview||!only||!ui.scene.objects.length;
  synLockOutputSplit();
}
// Only the split of the scene's sources can be chosen for saving.
function synLockOutputSplit() {
  const only=synOnlySplit(syntheticUI.preview);
  $('syn-output-split').querySelectorAll('option').forEach(option=>option.disabled=!!syntheticUI.preview&&option.value!==only);
}
async function synCommit(candidate,history=true) {
  const preview=await synPost('preview',{scene:candidate});
  const ui=syntheticUI;
  if(history) {ui.undo.push(synCopy(ui.scene));if(ui.undo.length>100)ui.undo.shift();ui.redo=[];}
  ui.scene=candidate;ui.preview=preview;ui.selected=Math.min(ui.selected,candidate.objects.length-1);
  synMessage('Сцена проверена: рамки, пиксели, границы и IoU корректны.');synDraw();
}
async function synHistory(redo) {
  const ui=syntheticUI,from=redo?ui.redo:ui.undo,to=redo?ui.undo:ui.redo;
  const candidate=from.at(-1);if(!candidate)return;
  const previous=synCopy(ui.scene);
  if(candidate.background)await synCommit(synCopy(candidate),false);
  else {ui.scene=synCopy(candidate);ui.preview=null;ui.selected=-1;synDraw();}
  from.pop();to.push(previous);ui.placing=false;
  if(synMounted()) {
    $('syn-bg-split').value=ui.scene.background?.split||'train';
    $('syn-bg-id').innerHTML='<option value="">Выберите фон</option>'+synOptions($('syn-bg-split').value,ui.scene.background?.id);
    $('syn-blend').value=ui.scene.blending;$('syn-iou').value=ui.scene.max_iou;
  }
  synDraw();
}
async function synPrepare(object=synActiveObject()) {
  if(!object)return;
  const ui=syntheticUI,token=++object.prepareToken;
  object.prepared=null;object.error=null;
  if(object===synActiveObject())ui.placing=false;
  synDraw();synButtons();
  try {
    const body={source:object.source,box_index:object.box_index,transform:object.transform};
    if(object.morph)body.morph=object.morph;
    const result=await synPost('prepare',body);
    if(token!==object.prepareToken)return;
    object.prepared=result;
    if(object===synActiveObject())synMessage('Объект готов. Разместите его на фоне кликом.');
    synDraw();synButtons();
  } catch(error) {
    if(token!==object.prepareToken)return;
    object.error=error.message;
    if(object===synActiveObject())synMessage(error.message,true);
    synDraw();synButtons();
  }
}
async function synLoadDonor() {
  const ui=syntheticUI,token=++ui.sourceToken;
  ui.donor=null;ui.placing=false;synDraw();synButtons();
  if(!ui.donorId)return;
  const donor=await request(`/api/synthetic/sample?split=${ui.donorSplit}&id=${encodeURIComponent(ui.donorId)}`);
  if(token!==ui.sourceToken)return;
  ui.donor=donor;synDraw();
}
function synPoint(event,canvas) {
  const rect=canvas.getBoundingClientRect(),s=synSensor();
  return {x:(event.clientX-rect.left)*s.frame_width/rect.width,y:(event.clientY-rect.top)*s.frame_height/rect.height};
}
function synHit(boxes,point) {
  for(let i=boxes.length-1;i>=0;i--) {const b=boxes[i];if(point.x>=b[0]&&point.x<=b[2]&&point.y>=b[1]&&point.y<=b[3])return i;}
  return -1;
}
function synNewObject(x,y) {
  const object=synActiveObject();
  const added={source:synCopy(object.source),box_index:object.box_index,transform:synCopy(object.transform),x,y};
  if(object.morph)added.morph=synCopy(object.morph);
  return added;
}
async function synAdd(x,y) {
  const ui=syntheticUI;if(!synActiveObject()?.prepared||!ui.preview)return;
  const candidate=synCopy(ui.scene);candidate.objects.push(synNewObject(x,y));
  await synCommit(candidate);ui.selected=candidate.objects.length-1;ui.placing=false;synDraw();
}
function synBind() {
  const ui=syntheticUI;
  $('syn-source-split').onchange=()=>synAction(async()=>{ui.donorSplit=$('syn-source-split').value;ui.donorId=null;$('syn-source-id').innerHTML='<option value="">Выберите изображение</option>'+synOptions(ui.donorSplit,null);await synLoadDonor();});
  $('syn-source-id').onchange=()=>synAction(async()=>{ui.donorId=$('syn-source-id').value;await synLoadDonor();});
  $('syn-source').onclick=event=>{
    if(ui.busy||!ui.donor)return;
    const index=synHit(ui.donor.boxes_pixels_xyxy,synPoint(event,$('syn-source')));
    if(index>=0)return synCollectObject(index);
  };
  for(const [id,key] of synTransformFields) {
    $(id).oninput=()=>{
      const object=synActiveObject();if(!object||ui.busy)return;
      object.transform[key]=$(id).type==='checkbox'?$(id).checked:Number($(id).value);
      return synPrepare(object);
    };
  }
  $('syn-reset').onclick=()=>{
    const object=synActiveObject();if(!object||ui.busy)return;
    object.transform=synDefaultTransform();synSyncTransformFields();return synPrepare(object);
  };
  $('syn-bg-split').onchange=()=>{$('syn-bg-id').innerHTML='<option value="">Выберите фон</option>'+synOptions($('syn-bg-split').value,null);};
  $('syn-bg-id').onchange=()=>synAction(async()=>{
    if(!$('syn-bg-id').value)return;
    const candidate=synCopy(ui.scene);candidate.background={split:$('syn-bg-split').value,id:$('syn-bg-id').value};
    try {await synCommit(candidate);} finally {if(synMounted()) {$('syn-bg-split').value=ui.scene.background?.split||'train';$('syn-bg-id').innerHTML='<option value="">Выберите фон</option>'+synOptions($('syn-bg-split').value,ui.scene.background?.id);}}
  });
  $('syn-bg-seed').onchange=()=>ui.backgroundSeed=Number($('syn-bg-seed').value);
  $('syn-bg-generate').onclick=()=>synAction(async()=>{
    const candidate=synCopy(ui.scene);
    candidate.background={split:$('syn-bg-split').value,generated:{seed:ui.backgroundSeed}};
    synMessage('Сборка фона… Первый раз читаются все кадры набора.');
    await synCommit(candidate);
    // The next click gives a new background; this one stays reproducible by its seed.
    ui.backgroundSeed+=1;if(synMounted()) {$('syn-bg-seed').value=ui.backgroundSeed;$('syn-bg-id').value='';}
  });
  for(const [id,key] of [['syn-blend','blending'],['syn-iou','max_iou']]) $(id).onchange=()=>synAction(async()=>{
    const candidate=synCopy(ui.scene);candidate[key]=id==='syn-iou'?Number($(id).value):$(id).value;
    try {if(candidate.background)await synCommit(candidate);else ui.scene=candidate;} finally {if(synMounted())$(id).value=ui.scene[key];}
  });
  $('syn-place').onclick=()=>{ui.placing=!ui.placing;ui.selected=-1;synMessage('Кликните центр нового объекта на фоне.');synDraw();synButtons();};
  $('syn-undo').onclick=()=>synAction(()=>synHistory(false));$('syn-redo').onclick=()=>synAction(()=>synHistory(true));
  $('syn-delete').onclick=()=>synAction(async()=>{if(ui.selected<0)return;const candidate=synCopy(ui.scene);candidate.objects.splice(ui.selected,1);await synCommit(candidate);ui.selected=-1;synDraw();});
  $('syn-position').onclick=()=>synAction(async()=>{const candidate=synCopy(ui.scene);candidate.objects[ui.selected].x=Number($('syn-x').value);candidate.objects[ui.selected].y=Number($('syn-y').value);await synCommit(candidate);});
  const canvas=$('syn-scene');
  canvas.onpointerdown=event=>{
    if(ui.busy||!ui.preview)return;
    const point=synPoint(event,canvas);
    if(ui.placing) {
      const prepared=synActiveObject()?.prepared;if(!prepared)return;
      synAction(()=>synAdd(Math.round(point.x-prepared.width/2),Math.round(point.y-prepared.height/2)));return;
    }
    const original=ui.preview.statistics.total_objects-ui.scene.objects.length;
    const index=synHit(ui.preview.boxes_pixels_xyxy.slice(original),point);ui.selected=index;synDraw();synButtons();
    if(index<0)return;
    const obj=ui.scene.objects[index];ui.drag={index,dx:point.x-obj.x,dy:point.y-obj.y,x:obj.x,y:obj.y};canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove=event=>{
    if(!ui.drag||ui.busy)return;
    const point=synPoint(event,canvas),drag=ui.drag;drag.x=Math.round(point.x-drag.dx);drag.y=Math.round(point.y-drag.dy);
    synDraw();const original=ui.preview.statistics.total_objects-ui.scene.objects.length,box=ui.preview.boxes_pixels_xyxy[original+drag.index];
    const ctx=canvas.getContext('2d');ctx.strokeStyle='#fff177';ctx.lineWidth=3;ctx.setLineDash([8,5]);
    ctx.strokeRect(drag.x*16,drag.y*16,(box[2]-box[0])*16,(box[3]-box[1])*16);ctx.setLineDash([]);
    $('syn-x').value=drag.x;$('syn-y').value=drag.y;
  };
  canvas.onpointerup=()=>{
    if(!ui.drag)return;const drag=ui.drag;ui.drag=null;
    if(ui.scene.objects[drag.index].x===drag.x&&ui.scene.objects[drag.index].y===drag.y)return;
    synAction(async()=>{const candidate=synCopy(ui.scene);candidate.objects[drag.index].x=drag.x;candidate.objects[drag.index].y=drag.y;try {await synCommit(candidate);} finally {synDraw();}});
  };
  canvas.onpointercancel=()=>{ui.drag=null;synDraw();};
  $('syn-directory').onchange=()=>ui.directory=$('syn-directory').value;
  $('syn-output-split').onchange=()=>ui.outputSplit=$('syn-output-split').value;
  for(const [id,key] of [['syn-count','count'],['syn-seed','seed']]) $(id).onchange=()=>ui[key]=Number($(id).value);
  for(const key of ['position','brightness','angle','scale']) $(`syn-var-${key}`).onchange=()=>ui.variation[key]=Number($(`syn-var-${key}`).value);
  const save=variations=>synAction(async()=>{
    ui.directory=$('syn-directory').value;ui.outputSplit=$('syn-output-split').value;
    synMessage(variations?'Генерация вариаций…':'Проверка и сохранение…');
    const result=await synPost(variations?'variations':'save',{scene:ui.scene,expected_digest:ui.preview.digest,directory:ui.directory,split:ui.outputSplit,count:ui.count,seed:ui.seed,variation:ui.variation});
    ui.output=variations?`Сохранено ${result.saved.length} / ${result.requested}.\n`+result.saved.map(item=>item.image).join('\n')+'\n'+result.skipped.map(item=>`Пропущена ${item.variation}: ${item.reason}`).join('\n'):`Сохранено:\n${result.image}\n${result.annotations}`;
    if(synMounted())$('syn-output').textContent=ui.output;synMessage(variations?`Сохранено ${result.saved.length}; пропущено ${result.skipped.length}.`:'Сцена сохранена: 8-битный PNG и YOLO-разметка.');
  });
  $('syn-save').onclick=()=>save(false);$('syn-variations').onclick=()=>save(true);
}
