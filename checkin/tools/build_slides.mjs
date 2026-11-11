// Editable mentor deck. Uses the same Artifact Tool facades as Artifact Session.
export function buildSlides(presentation, session) {
  presentation.slides.keep([]);
  const C={ink:'#172B4D',muted:'#56616F',teal:'#087F8C'};
  const text=(s,name,value,x,y,w,h,size=28,bold=false,color=C.ink)=>{
    const a=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    a.text=value;a.text.style={fontFamily:'Arial',fontSize:size,bold,color,verticalAlignment:'middle'};return a;
  };
  const add=()=>presentation.slides.add({width:1280,height:720,background:{fill:'#FFFFFF'}});
  const s1=add();
  text(s1,'project-title','A Transformer for Stock\nPrediction in Needle',72,44,1140,148,60,true);
  text(s1,'goal','Learn from a stock’s history and from other stocks.',72,220,1120,58,30,false,C.muted);
  text(s1,'temporal-label','Temporal attention',72,332,390,58,34,true);
  text(s1,'temporal-definition','Relationships across dates\nwithin the same stock',520,323,660,90,30);
  text(s1,'stock-label','Cross-stock attention',72,445,410,58,34,true);
  text(s1,'stock-definition','Relationships between stocks\non the same date',520,436,660,90,30);
  text(s1,'implementation-goal','Needle modules + custom CUDA attention',72,588,1110,54,30,true,C.teal);
  const talk1='Our project is a Transformer for stock prediction, implemented in Needle. We use two types of attention. Temporal attention looks at different dates in the history of one stock. Cross-stock attention allows the model to use information from other stocks on the same date. The goal is to implement these components in Needle and study a GPU attention operation that uses less memory.';
  s1.speakerNotes.text=talk1+'\n\n中文提示：先讲两种 attention 的区别，不假设老师了解金融。\nSources: Li et al., MASTER, AAAI 2024, https://doi.org/10.1609/aaai.v38i1.27767 ; local master_needle/model.py. The overview omits market gating and pooling details. Architecture is inspired by MASTER, not claimed as a new forecasting architecture.';
  const s2=add();
  text(s2,'implementation-title','Implementation and first checks',72,52,1130,75,44,true);
  text(s2,'prototype-status','A working prototype, with a short runnable demo',72,152,1120,56,30,false,C.muted);
  text(s2,'built-heading','What we built',72,257,540,50,32,true);
  text(s2,'checked-heading','What we check',704,257,500,50,32,true);
  text(s2,'built-one','Temporal + cross-stock attention',72,336,586,55,28);
  text(s2,'built-two','CUDA forward and backward',72,414,580,55,28);
  text(s2,'built-three','Attention computed in blocks',72,492,580,55,28);
  text(s2,'check-one','Outputs and gradients',704,336,500,55,28);
  text(s2,'check-two','Small model forward pass',704,414,500,55,28);
  text(s2,'check-three','Short learning run',704,492,500,55,28);
  text(s2,'demo-launch','Short demo: Run_Demo.cmd',72,609,1100,46,27,true,C.teal);
  const talk2='We now have a working prototype with temporal and cross-stock attention. For the GPU attention operation, we process the computation in blocks and recompute intermediate values during backward. This reduces the attention values that must be saved. We have run initial output and gradient checks, as well as small training experiments. The short demo checks the attention outputs and gradients, runs the model on a small input, and shows a short learning run. The synthetic loss is only a learning check.';
  s2.speakerNotes.text=talk2+'\n\n中文提示：此处可切到短测试。看到 ALL CHECKS PASSED 后回到 slides；不要现场跑完整训练。\nSources: checkin/demo_checkin.py ; master_needle/kernels/attention.cu ; Dao et al., FlashAttention, 2022, https://arxiv.org/abs/2205.14135 . CuPy provides basic arrays/cuBLAS. Attention kernels and model/autodiff integration are in Needle; PyTorch is an independent test reference only.';
  const s3=add();
  text(s3,'results-title','Initial results and remaining work',72,52,1130,75,44,true);
  text(s3,'memory-heading','Memory used in one test',72,166,570,55,30,true);
  const chart=s3.charts.add('bar',{
    position:{left:72,top:236,width:555,height:314},categories:['Dense attention','Block attention'],
    series:[{name:'Peak allocation (MiB)',values:[397.0,158.7],fill:C.teal}],hasLegend:false,
    barOptions:{direction:'column',grouping:'clustered',gapWidth:140},chartFill:'none',plotAreaFill:'none',
    dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:23,fill:C.ink}},
    xAxis:{textStyle:{fontSize:21,fill:C.muted}},
    yAxis:{min:0,max:450,majorUnit:100,numberFormatCode:'0',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{style:'solid',fill:'#E0E5E9',width:1}}
  });
  for(const style of [chart.legend?.textStyle,chart.dataLabels?.textStyle,chart.xAxis?.textStyle,chart.yAxis?.textStyle])if(style)style.typeface='Arial';
  text(s3,'memory-unit','Peak live GPU array allocations (MiB)',72,558,565,40,20,false,C.muted);
  text(s3,'tradeoff-heading','Memory–time trade-off',700,166,505,55,30,true);
  text(s3,'latency','Training step: 15.7 → 21.8 ms',700,246,500,65,27);
  text(s3,'tradeoff-note','Lower memory, but slower\nin this test.',700,316,500,85,28,false,C.muted);
  text(s3,'next-heading','Remaining work',700,427,500,45,30,true);
  text(s3,'next-list','Review FP32 training differences\nFinalize evaluation and report\nPrepare the 90-second video',700,488,520,127,25);
  text(s3,'measurement-note','1,024 stocks · 12 steps · FP32 · width 32. Memory excludes the CUDA context and external workspaces.',72,642,1135,40,17,false,C.muted);
  const talk3='Our initial tests show a memory and time trade-off. In one full-model test with 1,024 stocks and twelve historical steps, measured peak GPU array allocations fell from about 397 to 159 MiB. The training step increased from about 16 to 22 milliseconds. Our remaining work is to review the numerical differences seen in longer FP32 training, summarize the results clearly, and prepare the final notebook and video. We are keeping the current project scope.';
  s3.speakerNotes.text=talk3+'\n\n中文提示：这是一组初步测试，省内存但更慢；不称整卡显存或速度提升，也不声称长期预测完全等价。\nSource: results/gpu_benchmark.json, measured 2026-10-26; GPU_RESULTS.md. Dense baseline uses handwritten VJP; all modes use the same first-order driver. Two warmups and five timed iterations. Peak live CuPy-pool allocations exclude CUDA context and external library workspaces. Real training uses 128 stocks separately from this synthetic systems benchmark.';
  return {talks:[talk1,talk2,talk3],slideCount:3};
}
