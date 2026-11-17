/* Separate modal workflow: object bank -> candidates -> explicit selection -> export. */
'use strict';
const autogenUI={pool:[],poolSelected:new Set(),background:null,candidates:[],selected:new Set(),saved:new Set(),
  busy:false,initialized:false,directory:'synthetic_data',batchSplit:null,message:'',output:'',
  rules:{flip_x:true,flip_y:false,rotation_degrees:10,brightness_min:0.9,brightness_max:1.1,
    insertion:'mask',edges:'hard',blur_radius:0.8,count_min:1,count_max:3,images:12,seed:17,max_iou:0.15,blending:'replace',
    gamma_min:1,gamma_max:1,contrast_min:1,contrast_max:1,noise_max:0,gradient_max:0,shade_probability:0,shade_max:0.5,erode_max:0,
    split:'train',background_mode:'fixed',background_block_min:2,background_block_max:8,
    background_level_min:1,background_level_max:1,morph_reseed:true,
    unique_per_image:true,unique_source:true,balanced:true}};
const autogenFields=[['auto-angle','rotation_degrees'],['auto-bright-min','brightness_min'],['auto-bright-max','brightness_max'],
  ['auto-blur','blur_radius'],['auto-count-min','count_min'],['auto-count-max','count_max'],['auto-images','images'],
  ['auto-seed','seed'],['auto-iou','max_iou'],['auto-insertion','insertion'],['auto-edges','edges'],['auto-blend','blending'],
  ['auto-flip-x','flip_x'],['auto-flip-y','flip_y'],
  ['auto-gamma-min','gamma_min'],['auto-gamma-max','gamma_max'],['auto-contrast-min','contrast_min'],
  ['auto-contrast-max','contrast_max'],['auto-noise','noise_max'],['auto-gradient','gradient_max'],
  ['auto-shade-prob','shade_probability'],['auto-shade-max','shade_max'],['auto-erode','erode_max'],
  ['auto-split','split'],['auto-bg-mode','background_mode'],['auto-bg-block-min','background_block_min'],
  ['auto-bg-block-max','background_block_max'],['auto-bg-level-min','background_level_min'],['auto-bg-level-max','background_level_max'],
  ['auto-morph-reseed','morph_reseed'],['auto-unique-image','unique_per_image'],['auto-unique-source','unique_source'],
  ['auto-balanced','balanced']];

function autogenMessage(text,error=false) {
  autogenUI.message=text;
  if($('auto-message')) {$('auto-message').textContent=text;$('auto-message').classList.toggle('error',error);}
}
function openAutogenerator() {
  const ui=autogenUI,dialog=$('autogen-dialog');
  if(!ui.busy) {
    const previousCount=ui.pool.length;
    ui.pool=synBankEntries();
    for(let index=previousCount;index<ui.pool.length;index++)ui.poolSelected.add(index);
    ui.background=synCopy(syntheticUI.scene.background);
    if(!ui.initialized&&ui.background?.split)ui.rules.split=ui.background.split;
    if(!ui.initialized) {
      ui.rules.max_iou=syntheticUI.scene.max_iou;ui.rules.blending=syntheticUI.scene.blending;
      ui.directory=syntheticUI.directory;ui.initialized=!!ui.background&&!!ui.pool.length;
    }
    renderAutogenerator();
  }
  if(!dialog.open)dialog.showModal();
}
function autogenBackgroundInfo() {
  const ui=autogenUI,mode=ui.rules.background_mode,split=ui.rules.split;
  $('auto-background').textContent=mode==='fixed'?`Фон: ${synBackgroundLabel(ui.background)}`+(autogenBackgroundFits()?'':` — не из ${split}, выберите в редакторе фон из ${split}`):
    mode==='real'?`Фон: случайный реальный кадр из ${split} для каждой картинки (его рамки сохраняются).`:
    `Фон: новый сгенерированный для каждой картинки, каналы из ${split}.`;
}
// A batch takes everything from its split: the fixed background must belong to it.
const autogenBackgroundFits = () => autogenUI.rules.background_mode!=='fixed'||autogenUI.background?.split===autogenUI.rules.split;
// Pool entries the batch can use: those of its split that are not stubs cut by the frame edge.
const autogenUsable = object => object.source.split===autogenUI.rules.split&&!object.clipped;
function autogenPoolNote(object) {
  if(object.clipped)return 'обрубок: не используется';
  return object.source.split!==autogenUI.rules.split?`не из ${autogenUI.rules.split}: не используется`:'';
}
function autogenReadRules() {
  const ui=autogenUI;
  for(const [id,key] of autogenFields) {
    const node=$(id);
    ui.rules[key]=node.type==='checkbox'?node.checked:node.type==='number'?Number(node.value):node.value;
  }
}
function renderAutogenerator() {
  const ui=autogenUI,r=ui.rules;
  $('autogen-dialog').innerHTML=`<div class="autogen-heading"><h2 id="autogen-title">Автогенератор синтетических данных</h2><button id="auto-close" aria-label="Закрыть автогенератор">Закрыть</button></div>
    <p>Подготовьте набор объектов (реальные рамки и/или объекты из генератора объектов). Фон — выбранный в редакторе, случайный реальный кадр или новый сгенерированный для каждой картинки. Сначала просмотрите кандидатов и отметьте подходящие.</p>
    <div id="auto-message" class="synthetic-message" role="status" aria-live="polite"></div>
    <div class="controls"><span id="auto-background"></span><button id="auto-editor">Перейти в редактор</button></div>
    <div id="auto-pool" class="autogen-pool"></div>
    <div class="autogen-layout"><section class="card"><h3>Правила генерации</h3><div class="autogen-rules">
      <h4>Часть</h4>
      <label>Часть набора <select id="auto-split">${synSplitOptions()}</select></label>
      <p class="muted">Объекты, фон (реальный кадр или пустые каналы для нового) и сохранение — только из этой части. Объекты набора из других частей и обрубки не используются.</p>
      <h4>Фон</h4>
      <label>Фон картинки <select id="auto-bg-mode"><option value="fixed">Выбранный в редакторе</option><option value="real">Случайный реальный кадр</option><option value="generated">Новый сгенерированный</option></select></label>
      ${synNumber('auto-bg-block-min','Каналов в блоке от',r.background_block_min,1,synSensor().channels,1)}
      ${synNumber('auto-bg-block-max','Каналов в блоке до',r.background_block_max,1,synSensor().channels,1)}
      ${synNumber('auto-bg-level-min','Уровень фона × от',r.background_level_min,0,3,0.05)}
      ${synNumber('auto-bg-level-max','Уровень фона × до',r.background_level_max,0,3,0.05)}
      <h4>Объекты</h4>
      <label>Без повторов объекта на картинке <input id="auto-unique-image" type="checkbox" ${r.unique_per_image?'checked':''}></label>
      <label>Не брать объекты одного источника и из кадра-фона <input id="auto-unique-source" type="checkbox" ${r.unique_source?'checked':''}></label>
      <label>Равномерно по набору (сначала реже использованные) <input id="auto-balanced" type="checkbox" ${r.balanced?'checked':''}></label>
      <label>Новые несовершенства на каждую вставку <input id="auto-morph-reseed" type="checkbox" ${r.morph_reseed?'checked':''}></label>
      <label>Случайное отражение по X <input id="auto-flip-x" type="checkbox" ${r.flip_x?'checked':''}></label>
      <label>Случайное отражение по Y <input id="auto-flip-y" type="checkbox" ${r.flip_y?'checked':''}></label>
      ${synNumber('auto-angle','Поворот ± °',r.rotation_degrees,0,180,1)}
      ${synNumber('auto-bright-min','Яркость × от',r.brightness_min,0,4,0.05)}
      ${synNumber('auto-bright-max','Яркость × до',r.brightness_max,0,4,0.05)}
      ${synNumber('auto-gamma-min','Гамма от',r.gamma_min,0.3,3,0.05)}
      ${synNumber('auto-gamma-max','Гамма до',r.gamma_max,0.3,3,0.05)}
      ${synNumber('auto-contrast-min','Контраст × от',r.contrast_min,0,3,0.05)}
      ${synNumber('auto-contrast-max','Контраст × до',r.contrast_max,0,3,0.05)}
      ${synNumber('auto-noise','Шум пикселей до',r.noise_max,0,1,0.01)}
      ${synNumber('auto-gradient','Градиент света ±',r.gradient_max,0,1,0.05)}
      ${synNumber('auto-shade-prob','Вероятность затенения',r.shade_probability,0,1,0.05)}
      ${synNumber('auto-shade-max','Макс. сила затенения',r.shade_max,0,1,0.05)}
      ${synNumber('auto-erode','Выедание краёв до',r.erode_max,0,1,0.05)}
      <label>Перенос <select id="auto-insertion"><option value="box">Весь бокс</option><option value="mask">Маска объекта</option></select></label>
      <label>Края вставки <select id="auto-edges"><option value="hard">Обычная вставка</option><option value="blur">Блюр краёв</option></select></label>
      ${synNumber('auto-blur','Радиус блюра, px',r.blur_radius,0.1,3,0.1)}
      <label>Наложение <select id="auto-blend"><option value="replace">Replace</option><option value="max">Max blending</option><option value="additive">Additive blending</option></select></label>
      ${synNumber('auto-count-min','Новых объектов от',r.count_min,1,128,1)}
      ${synNumber('auto-count-max','Новых объектов до',r.count_max,1,128,1)}
      ${synNumber('auto-images','Количество картинок',r.images,1,1000,1)}
      ${synNumber('auto-seed','Seed',r.seed,0,4294967295,1)}
      ${synNumber('auto-iou','Макс. IoU',r.max_iou,0,1,0.01)}
      <button id="auto-generate">Сгенерировать предпросмотры</button>
    </div><p>Яркость и поворот изменяются относительно настроек каждой записи. Порог маски и масштаб берутся из набора. Разрешённые отражения применяются с вероятностью 50%. Объекты выбираются случайно с учётом правил повторов: «один источник» — это одна реальная рамка, из которой сделан объект (в том числе сгенерированный); объекты из кадра, выбранного фоном, не берутся.</p></section>
    <section><h3>Результаты для отбора</h3><p id="auto-results-info"></p>
      <div class="controls"><button id="auto-select-all">Выбрать все</button><button id="auto-select-none">Снять выбор</button><span id="auto-selected-count"></span></div>
      <div id="auto-results" class="autogen-gallery"></div>
    </section></div><div class="autogen-save"><div class="synthetic-fields">
      <label class="path">Отдельная директория <input id="auto-directory" type="text" value="${esc(ui.directory)}"></label>
      <span id="auto-save-split"></span>
      <button id="auto-save">Сохранить выбранные</button></div><p id="auto-provenance"></p><output id="auto-output" class="synthetic-output"></output></div>`;
  autogenBackgroundInfo();
  autogenPool();
  for(const [id,key] of autogenFields) {
    const node=$(id);
    if(node.type==='checkbox')node.checked=r[key];else node.value=r[key];
    node.onchange=()=>{autogenReadRules();autogenBackgroundInfo();if(id==='auto-split')autogenPool();autogenControls();};
  }
  $('auto-close').onclick=()=>$('autogen-dialog').close();
  $('auto-editor').onclick=()=>{$('autogen-dialog').close();renderSyntheticEditor();};
  $('auto-generate').onclick=()=>autogenAction(generateAutogenPreviews);
  $('auto-save').onclick=()=>autogenAction(saveAutogenSelection);
  $('auto-select-all').onclick=()=>{ui.selected=new Set(ui.candidates.map((_,index)=>index).filter(index=>!ui.saved.has(index)));autogenSelection();};
  $('auto-select-none').onclick=()=>{ui.selected.clear();autogenSelection();};
  $('auto-directory').onchange=()=>ui.directory=$('auto-directory').value;
  $('auto-output').textContent=ui.output;
  autogenMessage(ui.message||'Выберите правила и запустите генерацию.');autogenResults();autogenControls();
}
function autogenPool() {
  const ui=autogenUI;
  $('auto-pool').innerHTML=ui.pool.length?ui.pool.map((object,index)=>{
    const note=autogenPoolNote(object);
    return `<label class="${note?'muted':''}"><input id="auto-pool-${index}" type="checkbox" ${ui.poolSelected.has(index)&&!note?'checked':''} ${note?'disabled':''}> <span>${index+1} · ${esc(object.source.split+' / '+object.source.id)}<br>${esc(synObjectLabel(object))}${note?'<br>'+esc(note):''}</span></label>`;
  }).join(''):'<p class="muted">Набор объектов пуст. Добавьте рамки в редакторе или объекты из генератора объектов.</p>';
  ui.pool.forEach((_,index)=>$(`auto-pool-${index}`).onchange=()=>{
    if($(`auto-pool-${index}`).checked)ui.poolSelected.add(index);else ui.poolSelected.delete(index);
    autogenControls();
  });
}
const autogenChosenPool = () => [...autogenUI.poolSelected].sort((a,b)=>a-b).map(index=>autogenUI.pool[index]).filter(autogenUsable);
async function autogenAction(action) {
  const ui=autogenUI;if(ui.busy)return;
  ui.busy=true;autogenControls();
  try {await action();} catch(error) {autogenMessage(error.message,true);}
  finally {ui.busy=false;autogenControls();}
}
function autogenControls() {
  if(!$('auto-generate'))return;
  const ui=autogenUI;
  document.querySelectorAll('#autogen-dialog input, #autogen-dialog select, #autogen-dialog button').forEach(node=>node.disabled=ui.busy);
  $('auto-close').disabled=false;
  $('auto-generate').disabled=ui.busy||(ui.rules.background_mode==='fixed'&&!ui.background)||!autogenBackgroundFits()||!autogenChosenPool().length;
  ui.pool.forEach((object,index)=>$(`auto-pool-${index}`).disabled=ui.busy||!autogenUsable(object));
  for(const id of ['auto-bg-block-min','auto-bg-block-max','auto-bg-level-min','auto-bg-level-max'])
    $(id).disabled=ui.busy||ui.rules.background_mode!=='generated';
  $('auto-blur').disabled=ui.busy||ui.rules.edges!=='blur';
  $('auto-select-all').disabled=$('auto-select-none').disabled=ui.busy||!ui.candidates.length;
  autogenSelection();
}
function autogenResults() {
  const ui=autogenUI;
  $('auto-results').innerHTML=ui.candidates.length?ui.candidates.map((candidate,index)=>`<article id="auto-card-${index}" class="card autogen-candidate">
      <label><input id="auto-choose-${index}" type="checkbox"> Кандидат ${index+1}</label>
      <canvas id="auto-preview-${index}" ${synCanvasSize(400)} aria-label="Предпросмотр кандидата ${index+1}"></canvas>
      <p>Новых: ${candidate.preview.statistics.added_objects} · всего: ${candidate.preview.statistics.total_objects} · IoU: ${candidate.preview.statistics.max_iou.toFixed(3)}</p>
      <p>Источники ${esc((candidate.preview.splits||[]).join(', '))} · фон ${esc(synBackgroundLabel(candidate.scene.background))}</p><p id="auto-saved-${index}" class="muted"></p></article>`).join(''):'<p class="empty">После генерации здесь появятся изображения с актуальными рамками.</p>';
  ui.candidates.forEach((candidate,index)=>{
    const p=candidate.preview,original=p.statistics.total_objects-p.statistics.added_objects;
    synCanvas(`auto-preview-${index}`,p.matrix,p.boxes_pixels_xyxy,-1,original);
    $(`auto-choose-${index}`).onchange=()=>{if($(`auto-choose-${index}`).checked)ui.selected.add(index);else ui.selected.delete(index);autogenSelection();};
  });
  $('auto-results-info').textContent=ui.candidates.length?'Отметьте подходящие изображения. Картинки не сохраняются при генерации.':'Картинки не сохраняются при генерации.';
  autogenSelection();
}
function autogenSelection() {
  const ui=autogenUI;
  $('auto-save-split').textContent=ui.batchSplit?`Часть: ${ui.batchSplit}`:'';
  $('auto-provenance').textContent=ui.batchSplit?`Кандидаты собраны только из ${ui.batchSplit} и сохраняются только в ${ui.batchSplit}.`:'Партия сохраняется в ту часть, из которой собрана.';
  $('auto-selected-count').textContent=`Выбрано ${ui.selected.size} / ${ui.candidates.length}`;
  $('auto-save').disabled=ui.busy||!ui.selected.size||!ui.batchSplit;
  ui.candidates.forEach((_,index)=>{
    $(`auto-choose-${index}`).checked=ui.selected.has(index);
    $(`auto-choose-${index}`).disabled=ui.busy||ui.saved.has(index);
    $(`auto-card-${index}`).classList.toggle('chosen',ui.selected.has(index));
    $(`auto-saved-${index}`).textContent=ui.saved.has(index)?'Сохранено':'';
  });
}
async function generateAutogenPreviews() {
  const ui=autogenUI;autogenReadRules();
  const pool=autogenChosenPool().map(({clipped,...entry})=>entry);
  autogenMessage('Генерация и проверка кандидатов…');
  const result=await synPost('generate',{background:ui.background,pool,rules:ui.rules});
  ui.candidates=result.candidates;ui.batchSplit=result.rules.split;ui.selected.clear();ui.saved.clear();ui.output='';
  autogenResults();$('auto-output').textContent='';
  let message=`Создано ${result.candidates.length} / ${result.requested}. Выберите подходящие и сохраните их.`;
  const used=result.usage.filter((_,index)=>index<pool.length);
  if(used.length)message+=`\nКаждый объект набора использован от ${Math.min(...used)} до ${Math.max(...used)} раз (по всем кандидатам).`;
  if(result.notes?.length)message+='\n'+result.notes.join('\n');
  if(result.candidates.length<result.requested)message+=' Не удалось набрать полный набор в пределах бюджета размещения.';
  if(Object.keys(result.rejected).length)message+='\nПропуски: '+Object.entries(result.rejected).map(([reason,count])=>`${reason} (${count})`).join('; ');
  autogenMessage(message);
}
async function saveAutogenSelection() {
  const ui=autogenUI,indices=[...ui.selected].sort((a,b)=>a-b);
  ui.directory=$('auto-directory').value;
  const items=indices.map(index=>({scene:ui.candidates[index].scene,expected_digest:ui.candidates[index].preview.digest}));
  autogenMessage('Проверка и сохранение выбранных изображений…');
  const result=await synPost('generated-save',{items,directory:ui.directory,split:ui.batchSplit});
  for(const saved of result.saved) {const index=indices[saved.selection];ui.saved.add(index);ui.selected.delete(index);}
  ui.output=result.saved.map(item=>item.image).join('\n')+'\n'+result.skipped.map(item=>`Кандидат ${indices[item.selection]+1}: ${item.reason}`).join('\n');
  $('auto-output').textContent=ui.output;autogenSelection();
  autogenMessage(`Сохранено ${result.saved.length} / ${result.selected}; пропущено ${result.skipped.length}.`);
}
$('autogen-open').onclick=openAutogenerator;
